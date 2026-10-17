"""Reuse existing point predictors; bootstrap ridges and infer selected GNN ensembles."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from proteinmod.multiprotein_experiments import fit_targets, macro_score
from proteinmod.uncertainty import bootstrap_indices


def ridge_ensemble(features, rows, roles, alphas, members, seed, native=False):
    train = roles["train"]
    scaler = StandardScaler().fit(features[train])
    x = scaler.transform(features)
    y = np.array([r["target"] for r in rows]) if native else fit_targets(rows, train)

    def score(pred):
        if not native:
            return macro_score(rows, roles["validation"], pred)
        return -float(
            np.mean(
                [
                    np.sqrt(
                        np.mean(
                            (
                                pred[
                                    [
                                        j
                                        for j, i in enumerate(roles["validation"])
                                        if rows[i]["protein_id"] == p
                                    ]
                                ]
                                - y[[i for i in roles["validation"] if rows[i]["protein_id"] == p]]
                            )
                            ** 2
                        )
                    )
                    for p in sorted({rows[i]["protein_id"] for i in roles["validation"]})
                ]
            )
        )

    best = None
    candidates = []
    for alpha in sorted(alphas):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000).fit(x[train], y[train])
        value = score(model.predict(x[roles["validation"]]))
        candidates.append({"alpha": alpha, "validation_score": value})
        if best is None or value > best[0]:
            best = value, alpha, model
    models = [best[2]]
    for j in range(members):
        ix = bootstrap_indices(rows, train, seed + j)
        models.append(
            Ridge(alpha=best[1], solver="lsqr", tol=1e-8, max_iter=10000).fit(x[ix], y[ix])
        )
    state = {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coefficient": np.stack([m.coef_ for m in models]),
        "intercept": np.array([m.intercept_ for m in models]),
    }
    predictions = ridge_predict(features, state)
    return (
        predictions,
        {
            "alpha": best[1],
            "candidates": candidates,
            "validation_score": best[0],
            "bootstrap_seed": seed,
            "bootstrap_members": members,
            "bootstrap_unit": "genotypes resampled within training protein; fixed training-only scale/target transform",
            "score_type": "negative_native_protein_macro_rmse"
            if native
            else "protein_macro_spearman",
        },
        state,
    )


def ridge_predict(features, state):
    x = (features - state["mean"]) / state["scale"]
    return (x @ state["coefficient"].T + state["intercept"]).T


def native_affine(rank_predictions, native, train, alpha):
    ix = np.array([i for i in train if np.isfinite(native[i])], int)
    if len(ix) < 3 or alpha < 0:
        raise ValueError("Native head requires training-only compatible labels")
    result = []
    coefficients = []
    for point in rank_predictions:
        model = Ridge(alpha=alpha).fit(point[ix, None], native[ix])
        result.append(model.predict(point[:, None]))
        coefficients.append([float(model.coef_[0]), float(model.intercept_)])
    return np.array(result), {
        "coefficient_intercept": coefficients,
        "training_ids_count": len(ix),
        "unit": "kcal/mol",
        "convention": "positive_destabilizing",
        "interpretation": "Training-only affine native head; no held-out assay scale fitting",
    }


def predict_hybrid(groups, state, settings, mode, n, roles):
    import torch

    from proteinmod.hybrid import HybridRegressor

    first = next(iter(groups.values()))
    model = HybridRegressor(
        first["graph"],
        first["residues"].shape[-1],
        settings["dimension"],
        settings["heads"],
        settings["layers"],
        mode,
    )
    model.load_state_dict(state)
    model.eval()
    result = np.full(n, np.nan)
    with torch.no_grad():
        for group in groups.values():
            g = group["graph"]
            degree = np.maximum(np.bincount(g["edge_index"][1], minlength=len(g["static"])), 1)
            topology = (
                torch.as_tensor(g["static"]),
                torch.as_tensor(g["edge_index"][0]),
                torch.as_tensor(g["edge_index"][1]),
                torch.as_tensor(g["edge_features"]),
                torch.tensor(degree, dtype=torch.float32)[None, :, None],
            )
            positions = torch.tensor([r["reference_position"] - 1 for r in g["nodes"]])
            batches = []
            for indices in roles.values():
                selected = np.flatnonzero(np.isin(group["indices"], indices))
                batches.extend(
                    selected[j : j + settings["batch_size"]]
                    for j in range(0, len(selected), settings["batch_size"])
                )
            for local in batches:
                residues = torch.from_numpy(
                    np.array(group["residues"][group["token_indices"][local]], copy=True)
                )
                result[group["indices"][local]] = model(
                    residues, torch.as_tensor(group["dynamic"][local]), topology, positions
                ).numpy()
    if not np.isfinite(result).all():
        raise ValueError("Missing/incompatible graph prediction")
    return result
