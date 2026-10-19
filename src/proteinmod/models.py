"""Fixed sequence features and validation-only ridge selection."""

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

from proteinmod.data import AMINO_ACIDS


def one_hot(sequences):
    length = len(sequences[0])
    lookup = {letter: index for index, letter in enumerate(AMINO_ACIDS)}
    columns = [
        position * 20 + lookup[letter]
        for sequence in sequences
        for position, letter in enumerate(sequence)
    ]
    rows = np.repeat(np.arange(len(sequences)), length)
    return sparse.csr_matrix(
        (np.ones(len(columns)), (rows, columns)), shape=(len(sequences), length * 20)
    )


def metrics(measured, predicted):
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    prediction_errors = measured - predicted
    constant = len(measured) < 2 or np.ptp(measured) == 0 or np.ptp(predicted) == 0
    return {
        "spearman_rho": None if constant else float(spearmanr(measured, predicted).statistic),
        "pearson_r": None if constant else float(pearsonr(measured, predicted).statistic),
        "mae": float(np.mean(np.abs(prediction_errors))),
        "rmse": float(np.sqrt(np.mean(prediction_errors**2))),
    }


def fit_ridge(train_x, train_y, validation_x, validation_y, alphas, seed):
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alphas must be finite and positive")
    candidates = []
    best = None
    for alpha in sorted(set(alphas)):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000, random_state=seed).fit(
            train_x, train_y
        )
        validation_rmse = metrics(validation_y, model.predict(validation_x))["rmse"]
        candidates.append({"alpha": alpha, "validation_rmse": validation_rmse})
        if best is None or validation_rmse < best[0]:
            best = validation_rmse, model, alpha
    return best[1], {
        "alpha": best[2],
        "selection": "minimum validation RMSE; ties smallest alpha",
        "candidates": candidates,
        "fit_partition": "train only",
    }
