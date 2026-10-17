"""Function-head metadata conditioning with assay-local evaluation and train-only scaling."""

import copy

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from proteinmod.assay_metadata import AssayHead, AssayMetadata, MetadataVocabulary, training_shuffle
from proteinmod.models import metrics
from proteinmod.multiprotein_experiments import fit_targets


def evaluate(rows, indices, prediction):
    assays = {}
    for assay in sorted({rows[i]["assay_id"] for i in indices}):
        local = [j for j, i in enumerate(indices) if rows[i]["assay_id"] == assay]
        first = rows[indices[local[0]]]
        assays[assay] = {
            "protein_id": first["protein_id"],
            "family": first["family"],
            "n": len(local),
            "spearman": metrics(
                [rows[indices[j]]["target"] for j in local], np.asarray(prediction)[local]
            )["spearman_rho"],
        }
    proteins = {
        p: float(np.mean([v["spearman"] or 0 for v in assays.values() if v["protein_id"] == p]))
        for p in sorted({v["protein_id"] for v in assays.values()})
    }
    return {
        "assays": assays,
        "proteins": proteins,
        "macro": float(np.mean(list(proteins.values()))),
        "aggregation": "assay Spearman, then equal endpoint mean per protein, then equal protein mean",
    }


def scale_fit(x, train, floor=1e-6):
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    scaler = StandardScaler().fit(x[train])
    scaler.scale_[np.sqrt(scaler.var_) < floor] = 1
    return scaler


def ridge(x, rows, roles, alphas, floor=1e-6):
    train, val = roles["train"], roles["validation"]
    scaler = scale_fit(x, train, floor)
    z = scaler.transform(x)
    y = fit_targets(rows, train)
    best, candidates = None, []
    for alpha in sorted(alphas):
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("Positive alpha required")
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000).fit(z[train], y[train])
        score = evaluate(rows, val, np.clip(model.predict(z[val]), 0, 1))["macro"]
        candidates.append({"alpha": alpha, "validation_macro": score})
        if best is None or score > best[0]:
            best = score, alpha, model
    if best is None:
        raise ValueError("Empty alpha grid")
    return (
        np.clip(best[2].predict(z), 0, 1),
        {
            "alpha": best[1],
            "validation_macro": best[0],
            "candidates": candidates,
            "parameters": len(best[2].coef_) + 1,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficient": best[2].coef_.tolist(),
            "intercept": float(best[2].intercept_),
        },
        scaler,
        best[2],
    )


def group_ridge(x, rows, roles, alphas, key, floor=1e-6):
    fallback, selected, _, _ = ridge(x, rows, roles, alphas, floor)
    prediction = fallback.copy()
    records = {}
    for group in sorted({rows[i][key] for i in roles["train"]}):
        train = np.array([i for i in roles["train"] if rows[i][key] == group])
        validation = np.array([i for i in roles["validation"] if rows[i][key] == group])
        if not len(validation):
            validation = roles["validation"]
            grid = [selected["alpha"]]
        else:
            grid = alphas
        pred, record, _, _ = ridge(
            x, rows, {"train": train, "validation": validation, "test": roles["test"]}, grid, floor
        )
        ix = [i for i, r in enumerate(rows) if r[key] == group]
        prediction[ix] = pred[ix]
        records[group] = record
    return prediction, {
        "groups": records,
        "fallback": selected,
        "unknown_test_rows": int(sum(rows[i][key] not in records for i in roles["test"])),
        "parameters": selected["parameters"] + sum(v["parameters"] for v in records.values()),
    }


def neural(h, rows, roles, metadata, settings, seed, mode):
    if settings["device"] != "cpu" or mode not in (
        "zero",
        "structured",
        "with_assay_id",
        "shuffled_train",
        "protein_proxy",
    ):
        raise ValueError("Deterministic CPU and declared mode required")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    proteins = [r["protein_id"] for r in rows]
    original = [AssayMetadata.from_dict(metadata[r["assay_id"]]) for r in rows]
    vocab = MetadataVocabulary.fit(
        [original[i] for i in roles["train"]], [proteins[i] for i in roles["train"]]
    )
    context, mapping = (
        training_shuffle(original, proteins, roles["train"], seed)
        if mode == "shuffled_train"
        else (original, {})
    )
    categorical, numeric = vocab.transform(context, proteins)
    scaler = scale_fit(h, roles["train"], settings["scale_floor"])
    x = torch.tensor(scaler.transform(h), dtype=torch.float32)
    c, n = torch.tensor(categorical), torch.tensor(numeric)
    y = torch.tensor(fit_targets(rows, roles["train"]), dtype=torch.float32)
    model = AssayHead(h.shape[1], vocab, settings["hidden"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(seed)

    def predict(indices, missing=False):
        model.eval()
        with torch.no_grad():
            cat, num = (
                (torch.zeros_like(c[indices]), torch.zeros_like(n[indices]))
                if missing
                else (c[indices], n[indices])
            )
            return model(x[indices], cat, num, mode).double().numpy()

    best, stale, trace = None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = rng.permutation(roles["train"])
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            ix = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[ix], c[ix], n[ix], mode) - y[ix]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite function loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        score = evaluate(rows, roles["validation"], predict(roles["validation"]))["macro"]
        trace.append(
            {"epoch": epoch, "training_mse_online": total / len(order), "validation_macro": score}
        )
        if best is None or score > best[0]:
            best = score, epoch, copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    all_ix = np.arange(len(rows))
    prediction = predict(all_ix)
    selection = {
        "epoch": best[1],
        "validation_macro": best[0],
        "trace": trace,
        "shuffle_mapping": mapping,
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_unknown_categories": {
            field: int((categorical[roles["test"], j] == 0).sum())
            for j, field in enumerate(vocab.state["vocabularies"])
        },
        "train_macro": evaluate(rows, roles["train"], prediction[roles["train"]])["macro"],
    }
    state = {
        "schema": "m9-assay-head-v1",
        "mode": mode,
        "head": best[2],
        "vocabulary": vocab.state,
        "hidden": settings["hidden"],
        "representation_mean": scaler.mean_.tolist(),
        "representation_scale": scaler.scale_.tolist(),
        "target_semantics": "bounded [0,1] training-assay percentile score; not calibrated probability or native assay unit",
    }
    return prediction, selection, state, predict(all_ix, True)
