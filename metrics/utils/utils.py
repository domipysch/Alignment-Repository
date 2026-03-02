from typing import List, Dict
import numpy as np
import pandas as pd


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two 1D arrays.

    Args:
        a: First sample as a 1D array.
        b: Second sample as a 1D array.

    Returns:
        Cohen's d as a float (0.0 if pooled std is zero or either array is empty).
    """
    if a.size < 1 or b.size < 1:
        return 0.0
    na, nb = a.size, b.size
    ma, mb = a.mean(), b.mean()
    sa, sb = np.std(a), np.std(b)

    denom = ((na - 1) * (sa**2) + (nb - 1) * (sb**2)) / (na + nb - 2)
    pooled_sd = np.sqrt(denom) if denom > 0 else 0.0
    if pooled_sd == 0:
        return 0.0
    return float((ma - mb) / pooled_sd)


def compute_norm_per_vector(vectors_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the L2 norm of each row in a DataFrame.

    Args:
        vectors_df: DataFrame of shape (n_vectors, n_dims).

    Returns:
        1D array of shape (n_vectors,) with the L2 norm of each row.
    """
    norms = vectors_df.apply(lambda row: np.sqrt((row**2).sum()), axis=1)
    return norms.astype(float).to_numpy()


def compute_log_norm_per_vector(norm_vectors: np.ndarray) -> np.ndarray:
    """
    Apply log to an array of norms, mapping non-positive values to 0 instead of -inf.

    Args:
        norm_vectors: 1D array of non-negative norm values.

    Returns:
        1D array of log-transformed norms, with log(x <= 0) replaced by 0.
    """
    # map log(0) -> 0 to avoid -inf
    with np.errstate(divide="ignore"):
        log_arr = np.where(norm_vectors <= 0, 0.0, np.log(norm_vectors))
    return log_arr


def compute_vector_metrics(vectors_df: pd.DataFrame) -> Dict[str, float]:
    """
    For a GEP matrix, compute basic metrics per gene.
    Args:
        vectors_df: GEP matrix (genes x spots/cells) as pandas DataFrame
    Returns:
        Dict with basic statistics of vector norms.
    """
    arr = compute_norm_per_vector(vectors_df)
    log_arr = compute_log_norm_per_vector(arr)

    def safe_min(a):
        return float(np.min(a)) if a.size > 0 else 0.0

    def safe_max(a):
        return float(np.max(a)) if a.size > 0 else 0.0

    def safe_mean(a):
        return float(np.mean(a)) if a.size > 0 else 0.0

    def safe_median(a):
        return float(np.median(a)) if a.size > 0 else 0.0

    def safe_std(a):
        if a.size <= 1:
            return 0.0
        return float(np.std(a, ddof=1))

    return {
        "min_norm": safe_min(arr),
        "max_norm": safe_max(arr),
        "mean_norm": safe_mean(arr),
        "median_norm": safe_median(arr),
        "std_norm": safe_std(arr),
        "log_mean_norm": safe_mean(log_arr),
        "log_std_norm": safe_std(log_arr),
    }


def compute_basic_metrics_for_gene_groups(
    gep: pd.DataFrame,
    marker_genes: List[str],
    non_marker_genes: List[str],
    include_norm_values: bool = False,
) -> Dict[str, float]:
    """
    Args:
        gep: GEP matrix of shape (genes x spots/cells) as a pandas DataFrame,
             with gene IDs as the index and cell/spot IDs as column headers.
        marker_genes: List of marker gene names
        non_marker_genes: List of non-marker gene names
        include_norm_values: Whether to also return all norm values as lists
    Returns: Dict with basic statistics for both gene groups
    """

    # Get spot vectors per genes for marker and non-marker genes
    marker_vectors = gep.loc[gep.index.astype(str).str.strip().isin(marker_genes)]
    non_marker_vectors = gep.loc[
        gep.index.astype(str).str.strip().isin(non_marker_genes)
    ]

    # Compute basic metrics for both groups
    marker_metrics = compute_vector_metrics(marker_vectors)
    non_marker_metrics = compute_vector_metrics(non_marker_vectors)

    # Compute norms and log norms as numpy-arrays
    norms_marker = compute_norm_per_vector(marker_vectors)
    log_norms_marker = compute_log_norm_per_vector(norms_marker)
    norms_non_marker = compute_norm_per_vector(non_marker_vectors)
    log_norms_non_marker = compute_log_norm_per_vector(norms_non_marker)
    cohen_d_norm = cohens_d(norms_marker, norms_non_marker)
    cohen_d_log_norm = cohens_d(log_norms_marker, log_norms_non_marker)

    result_metrics: Dict[str, float] = {
        # counts
        "n_marker_vectors": len(norms_marker),
        "n_non_marker_vectors": len(norms_non_marker),
        # marker group metrics (prefix marker_)
        **{f"marker_{k}": float(v) for k, v in marker_metrics.items()},
        # non-marker group metrics (prefix non_marker_)
        **{f"non_marker_{k}": float(v) for k, v in non_marker_metrics.items()},
        # Cohen's d
        "cohen_d_norm": cohen_d_norm,
        "cohen_d_log_norm": cohen_d_log_norm,
    }

    if include_norm_values:
        result_metrics["marker_norms"] = norms_marker.tolist()
        result_metrics["non_marker_norms"] = norms_non_marker.tolist()

    return result_metrics
