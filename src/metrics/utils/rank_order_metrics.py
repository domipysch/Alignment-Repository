from scipy.stats import spearmanr, kendalltau, rankdata
import numpy as np


def spearman_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Spearman rank-order distance between two 1D arrays.

    Args:
        a: First 1D array.
        b: Second 1D array of the same shape.

    Returns:
        Distance in [0, 1], or nan if undefined.
    """
    assert a.shape == b.shape, "Not equal shape"
    assert a.ndim == 1, "Not 1D arrays"
    a_ranks = rankdata(a, method="average")
    b_ranks = rankdata(b, method="average")
    try:
        r, _ = spearmanr(a_ranks, b_ranks)
    except Exception:
        r = np.nan
    return float(np.nan if np.isnan(r) else 1.0 - abs(r))


def kendall_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Kendall rank-order distance between two 1D arrays.

    Args:
        a: First 1D array.
        b: Second 1D array of the same shape.

    Returns:
        Distance in [0, 1], or nan if undefined.
    """
    assert a.shape == b.shape, "Not equal shape"
    assert a.ndim == 1, "Not 1D arrays"
    a_ranks = rankdata(a, method="average")
    b_ranks = rankdata(b, method="average")
    try:
        tau, _ = kendalltau(a_ranks, b_ranks)
    except Exception:
        tau = np.nan
    return float(np.nan if np.isnan(tau) else 1.0 - abs(tau))
