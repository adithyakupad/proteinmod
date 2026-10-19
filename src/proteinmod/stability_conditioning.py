"""M8 scalar conditioning: train-only calibrators and label-blind negative controls."""

import copy
import hashlib

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from proteinmod.multiprotein_experiments import fit_targets, macro_score
from proteinmod.stability_experiments import scores, validate_task


def fit_stability(x, rows, roles, alphas):
    """M6 native-unit ridge, returning a fitted scorer rather than just test predictions."""
    validate_task(rows)
    scaler = StandardScaler().fit(x[roles["train"]])
    train, validation = roles["train"], roles["validation"]
    y = np.array([r["target"] for r in rows])
    best, candidates = None, []
    for alpha in sorted(alphas):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Positive ridge penalty required")
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000).fit(
            scaler.transform(x[train]), y[train]
        )
        score = scores(rows, validation, model.predict(scaler.transform(x[validation])))["macro"][
            "rmse"
        ]
        candidates.append({"alpha": alpha, "validation_rmse_kcal_mol": score})
        if best is None or score < best[0]:
            best = score, alpha, model
    if best is None:
        raise ValueError("Empty ridge grid")
    return (
        scaler,
        best[2],
        {"alpha": best[1], "validation_rmse_kcal_mol": best[0], "candidates": candidates},
    )


def controls(z, rows, roles, seed):
    """Never import values across protein/role boundaries or read outcome labels."""
    z = np.asarray(z, dtype=float)
    train = roles["train"]
    if not np.isfinite(z[train]).all():
        raise ValueError("Missing training stability predictions")
    rng = np.random.default_rng(seed)
    shuffled = z.copy()
    for indices in roles.values():
        for protein in sorted({rows[i]["protein_id"] for i in indices}):
            ix = np.array([i for i in indices if rows[i]["protein_id"] == protein])
            shuffled[ix] = z[rng.permutation(ix)]
    # IDs generate independent noise; only training noise statistics set its scale.
    noise = np.array(
        [
            np.random.default_rng(
                int.from_bytes(hashlib.sha256(f"{seed}:{r['id']}".encode()).digest()[:8], "big")
            ).normal()
            for r in rows
        ]
    )
    scale = float(z[train].std())
    dummy = (noise - noise[train].mean()) / max(float(noise[train].std()), 1e-8)
    dummy = dummy * scale + z[train].mean()
    dummy[~np.isfinite(z)] = np.nan
    return {"predicted": z.copy(), "shuffled": shuffled, "dummy": dummy}


def scalar_inputs(z, train, observed=None):
    """Missing predictions require a mask, and never become an observed zero ΔΔG."""
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(z) if observed is None else np.asarray(observed)
    if mask.dtype != bool or mask.shape != z.shape or z.ndim != 1:
        raise ValueError("Explicit scalar availability mask has invalid dimensions")
    if not np.isfinite(z[mask]).all() or not mask[train].any():
        raise ValueError("Nonfinite observed scalar or no training observations")
    mean = float(z[train][mask[train]].mean())
    scale = max(float(z[train][mask[train]].std()), 1e-8)
    values = np.zeros_like(z)
    values[mask] = (z[mask] - mean) / scale
    return np.column_stack((values, mask)), {"mean_kcal_mol": mean, "scale_kcal_mol": scale}


def ridge_function(x, rows, roles, alphas, base=None):
    train, validation = roles["train"], roles["validation"]
    indices = np.unique(np.concatenate(list(roles.values())))
    if not np.isfinite(x[indices]).all():
        raise ValueError("Nonfinite function features")
    target = fit_targets(rows, train)
    offset = np.zeros(len(rows)) if base is None else np.asarray(base)
    if not np.isfinite(offset[indices]).all():
        raise ValueError("Nonfinite fixed function baseline")
    scaler = StandardScaler().fit(x[train])
    best, candidates = None, []
    for alpha in sorted(alphas):
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("Positive ridge penalty required")
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000).fit(
            scaler.transform(x[train]), target[train] - offset[train]
        )
        pred = model.predict(scaler.transform(x[validation])) + offset[validation]
        score = macro_score(rows, validation, pred)
        candidates.append({"alpha": alpha, "validation_macro_spearman": score})
        if best is None or score > best[0]:
            best = score, alpha, model
    if best is None:
        raise ValueError("Empty ridge grid")
    pred = np.full(len(rows), np.nan)
    pred[indices] = best[2].predict(scaler.transform(x[indices])) + offset[indices]
    return pred, {
        "alpha": best[1],
        "validation_macro_spearman": best[0],
        "candidates": candidates,
        "coefficients": best[2].coef_.tolist(),
        "intercept": float(best[2].intercept_),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


class ConditionedHead(nn.Module):
    """Only this head trains; cached M7 representation and predicted stability are frozen."""

    def __init__(self, dimension, hidden=16):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dimension + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, representation, scalar):
        if (
            representation.ndim != 2
            or scalar.shape != (len(representation), 2)
            or not torch.isfinite(representation).all()
            or not torch.isfinite(scalar).all()
        ):
            raise ValueError("Invalid conditioned head tensors")
        return self.head(torch.cat((representation, scalar), -1)).squeeze(-1)


def neural_function(h, scalar, rows, roles, settings, seed):
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    train, validation = roles["train"], roles["validation"]
    evaluation_indices = np.unique(np.concatenate(list(roles.values())))
    if (
        min(settings[k] for k in ("cpu_threads", "hidden", "max_epochs", "patience", "batch_size"))
        < 1
    ):
        raise ValueError("Positive head training settings required")
    scaler = StandardScaler().fit(h[train])
    representation_inputs = torch.tensor(scaler.transform(h), dtype=torch.float32)
    stability_inputs = torch.tensor(scalar, dtype=torch.float32)
    target = fit_targets(rows, train)
    target_mean, target_scale = float(target[train].mean()), max(float(target[train].std()), 1e-8)
    scaled_targets = torch.tensor((target - target_mean) / target_scale, dtype=torch.float32)
    model = ConditionedHead(h.shape[1], settings["hidden"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    random_generator = np.random.default_rng(seed)

    def predict(indices):
        model.eval()
        with torch.no_grad():
            return (
                model(representation_inputs[indices], stability_inputs[indices]).double().numpy()
                * target_scale
                + target_mean
            )

    best, stale_epochs, trace = None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = random_generator.permutation(train)
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = (
                (
                    model(representation_inputs[batch], stability_inputs[batch])
                    - scaled_targets[batch]
                )
                .square()
                .mean()
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite function loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        validation_score = macro_score(rows, validation, predict(validation))
        trace.append({"epoch": epoch, "validation_macro_spearman": validation_score})
        if best is None or validation_score > best[0]:
            best = validation_score, epoch, copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    prediction = np.full(len(rows), np.nan)
    prediction[evaluation_indices] = predict(evaluation_indices)
    state = {
        "schema": "m8-conditioned-head-v1",
        "head": best[2],
        "representation_mean": scaler.mean_.tolist(),
        "representation_scale": scaler.scale_.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "function_unit": "training-assay percentile",
        "stability_input_unit": "kcal/mol",
        "stability_positive_means": "destabilization",
    }
    return (
        prediction,
        {
            "epoch": best[1],
            "validation_macro_spearman": best[0],
            "trace": trace,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        state,
    )
