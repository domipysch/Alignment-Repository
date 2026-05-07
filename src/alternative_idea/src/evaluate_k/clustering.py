"""Unsupervised clustering metrics and Leiden reference clustering."""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import _dense_X, run_pca_neighbors_umap

logger = logging.getLogger(__name__)


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    sample = min(5_000, len(X))
    return float(
        silhouette_score(
            X, labels, metric="euclidean", sample_size=sample, random_state=0
        )
    )


def compute_dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Approximation: uses centroid distances for inter-cluster and
    max distance-to-centroid for intra-cluster diameter (O(N·K)).
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")

    centroids = np.array([X[labels == k].mean(axis=0) for k in unique])

    intra_diameters = []
    for i, k in enumerate(unique):
        pts = X[labels == k]
        dists = np.linalg.norm(pts - centroids[i], axis=1)
        intra_diameters.append(float(dists.max()) if len(pts) > 1 else 0.0)

    inter_dists = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
    ]

    max_intra = max(intra_diameters) if intra_diameters else 1.0
    min_inter = min(inter_dists) if inter_dists else 0.0
    return min_inter / max_intra if max_intra > 0 else float("nan")


def compute_modularity(adata_processed: AnnData, labels: np.ndarray) -> float:
    """
    Modularity of the given partition on the precomputed scanpy KNN graph
    (adata.obsp['connectivities']).  Requires igraph (bundled with scanpy).
    """
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping modularity")
        return float("nan")

    if "connectivities" not in adata_processed.obsp:
        logger.warning("No precomputed neighbors graph — skipping modularity")
        return float("nan")

    A = adata_processed.obsp["connectivities"].tocoo()
    edges = list(zip(A.row.tolist(), A.col.tolist()))
    g = ig.Graph(n=adata_processed.n_obs, edges=edges, directed=False)
    g.simplify()
    return float(g.modularity(labels.tolist()))


def compute_centroid_cosim(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity of per-state centroids.
    Only upper-triangle pairs (no self-similarity, no double-counting).
    Higher is better (states share similar expression profiles).
    """
    unique = np.unique(labels)
    K = len(unique)
    if K < 2:
        return float("nan")

    centroids = np.array([X[labels == k].mean(axis=0) for k in unique])
    sims = cosine_similarity(centroids)  # (K, K)
    idx = np.triu_indices(K, k=1)
    return float(sims[idx].mean())


def compute_all_metrics(
    adata_processed: AnnData,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute all four clustering quality metrics.

    Parameters
    ----------
    adata_processed : AnnData with PCA, neighbors, and UMAP already computed.
    labels          : Integer cluster assignment for each cell.
    """
    X = _dense_X(adata_processed)
    return {
        "silhouette": compute_silhouette(X, labels),
        "dunn_index": compute_dunn_index(X, labels),
        "modularity": compute_modularity(adata_processed, labels),
        "centroid_cosim": compute_centroid_cosim(X, labels),
    }


def run_leiden_shared_genes(
    adata_processed: AnnData,
    shared_genes: list[str],
    resolution: float,
) -> tuple[np.ndarray, AnnData]:
    """
    Standalone Leiden clustering using only the sc/st shared genes.

    Subsets adata_processed to shared_genes, recomputes PCA/neighbors/UMAP
    on that subset, then runs Leiden.

    Returns
    -------
    labels         : Integer cluster labels, shape (n_cells,).
    adata_shared   : AnnData subset (shared genes only) with UMAP + Leiden.
    """
    available = [g for g in shared_genes if g in adata_processed.var_names]
    if len(available) < 2:
        raise ValueError(
            f"Too few shared genes found in adata ({len(available)}); cannot cluster."
        )
    logger.info("Leiden on shared genes: %d genes", len(available))

    adata_shared = adata_processed[:, available].copy()
    run_pca_neighbors_umap(adata_shared)
    sc.tl.leiden(adata_shared, resolution=resolution, key_added="_leiden_shared")
    labels = adata_shared.obs["_leiden_shared"].astype(int).values
    logger.info(
        "Leiden on shared genes (resolution=%.2f): %d clusters",
        resolution,
        len(np.unique(labels)),
    )
    return labels, adata_shared


def run_leiden_clustering(
    adata_sc: AnnData,
    resolution: float,
) -> tuple[np.ndarray, AnnData]:
    """
    Standalone Leiden clustering on the sc data.

    Returns
    -------
    labels          : Integer cluster labels, shape (n_cells,).
    adata_processed : Working copy of adata_sc with UMAP + Leiden stored.
    """
    adata = adata_sc.copy()
    run_pca_neighbors_umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="_leiden_ref")
    labels = adata.obs["_leiden_ref"].astype(int).values
    logger.info(
        "Leiden clustering (resolution=%.2f): %d clusters",
        resolution,
        len(np.unique(labels)),
    )
    return labels, adata
