"""Shared low-level helpers used across the evaluate_k package."""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData

logger = logging.getLogger(__name__)


def _to_numpy(matrix: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _dense_X(adata: AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def _prepare_for_umap(adata_sc: AnnData, normalize: bool = True) -> AnnData:
    """
    Return a working copy of adata_sc with UMAP coordinates.
    Runs normalize → log1p → HVG → PCA → neighbors → UMAP
    if each step is not already present.
    """
    adata = adata_sc.copy()

    if "X_umap" in adata.obsm:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
        return adata

    logger.info("Preparing sc data for UMAP (normalize=%s)…", normalize)

    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if "X_pca" not in adata.obsm:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=2000, flavor="seurat", inplace=True
        )
        sc.pp.pca(adata, n_comps=30, use_highly_variable=True)

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")

    sc.tl.umap(adata)
    return adata


def hard_assignments(matrix: torch.Tensor | np.ndarray) -> np.ndarray:
    """Row-wise argmax → shape (N,)."""
    return _to_numpy(matrix).argmax(axis=1)
