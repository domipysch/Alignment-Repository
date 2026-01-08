import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from scipy.spatial.distance import braycurtis, jensenshannon
from scipy.stats import pearsonr


def getis_ord_g_stat(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the Getis-Ord G* statistic for each location i.

    Parameters
    ----------
    x : np.ndarray
        1D array of attribute values (length n).
    W : np.ndarray
        2D spatial-weights matrix (n × n), where W[i,j] = w_ij.

    See https://en.wikipedia.org/wiki/Getis%E2%80%93Ord_statistics.

    Returns
    -------
    np.ndarray
        Array of G*_i values (length n).
    """

    # denominator: sum_j x_j
    denom = x.sum()

    if denom == 0:
        return np.full(x.shape, np.nan)

    # numerator for each i: sum_j w_ij * x_j
    numer = W @ x

    # elementwise division
    return np.array(numer / denom)


def _to_prob_vector(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convert v to have sum=1
    """
    assert v.ndim == 1, "Not 1D arrays"
    assert min(v) >= 0
    s = v.sum()
    if s == 0.0:
        n = v.size if v.size > 0 else 1
        return np.full(v.shape, 1.0 / n, dtype=float)
    arr = v / s
    arr = np.maximum(arr, eps)
    arr = arr / arr.sum()
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape, "Not equal shape"
    assert a.ndim == 1, "Not 1D arrays"
    a = np.asarray(a).ravel().reshape(1, -1)
    b = np.asarray(b).ravel().reshape(1, -1)
    cossim = float(sklearn_cosine_similarity(a, b)[0, 0])
    # sometimes rounding issues occur: 1.00000000001, then problem with e.g. sqrt
    return min(cossim, 1.0)


def sqrt_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    cs = cosine_similarity(a, b)
    return float(np.sqrt(1 - cs))


def euclidean_l2(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def canberra(a: np.ndarray, b: np.ndarray) -> float:
    """Canberra distance (sum |x-y| / (|x|+|y|))"""
    denom = np.abs(a) + np.abs(b)
    diff = np.abs(a - b)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(denom == 0, 0.0, diff / denom)
    return float(np.sum(terms))


def pearson_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Pearson correlation coefficient (returns nan as float if undefined)"""
    if not np.any(a) or not np.any(b):
        return float(np.nan)
    try:
        r, _ = pearsonr(a, b)
    except Exception:
        r = np.nan
    return float(np.nan if np.isnan(r) else 1.0 - abs(r))


def bray_curtis_distance(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(a) and not np.any(b):
        return float(np.nan)
    return float(braycurtis(a, b))


def aitchison_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    p, q = _to_prob_vector(a, eps=eps), _to_prob_vector(b, eps=eps)
    clr_p = np.log(p) - np.log(p).mean()
    clr_q = np.log(q) - np.log(q).mean()
    return float(np.linalg.norm(clr_p - clr_q))


def kl_divergence(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    p, q = _to_prob_vector(a, eps=eps), _to_prob_vector(b, eps=eps)
    return float(np.sum(p * np.log(p / q)))


def jensen_shannon_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen–Shannon distance = sqrt(JS divergence)"""
    p, q = _to_prob_vector(a, eps=eps), _to_prob_vector(b, eps=eps)
    return jensenshannon(p, q)


def hellinger_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Hellinger distance between two probability distributions"""
    p, q = _to_prob_vector(a, eps=eps), _to_prob_vector(b, eps=eps)
    return float((1.0 / np.sqrt(2.0)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q)))


def bhattacharyya_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Bhattacharyya distance = -ln(sum sqrt(p*q))"""
    p, q = _to_prob_vector(a, eps=eps), _to_prob_vector(b, eps=eps)
    bc = float(np.sum(np.sqrt(p * q)))
    return bc


def total_variation(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """Total variation distance = 0.5 * L1 over probability vectors"""
    p, q = _to_prob_vector(a, eps=eps), _to_prob_vector(b, eps=eps)
    return float(0.5 * np.sum(np.abs(p - q)))


def smape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error between two 1D arrays.
    sMAPE = mean( 2 * |a - b| / (|a| + |b|) )
    Returns a float in [0, 2]. Small eps added to denominator for numerical stability.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("Inputs must have the same shape for smape.")
    denom = np.abs(a) + np.abs(b)
    # avoid zero division: where denom == 0, define contribution as 0 (since a==b==0 -> no error)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 2.0 * np.abs(a - b) / (denom + eps)
    # ensure finite
    frac[~np.isfinite(frac)] = 0.0
    return float(np.mean(frac))
