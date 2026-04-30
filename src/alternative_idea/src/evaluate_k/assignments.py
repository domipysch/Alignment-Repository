"""Cell-state assignment outputs: CSV files, fractions, and state AnnData."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import _dense_X

logger = logging.getLogger(__name__)


def save_cell_mapping_csv(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    output_path: Path,
) -> None:
    pd.DataFrame({"cellID": adata_sc.obs_names, "cell_state": cell_states}).to_csv(
        output_path, index=False
    )
    logger.info("Cell mapping → %s", output_path)


def save_spot_mapping_csv(
    adata_st: AnnData,
    spot_states: np.ndarray,
    output_path: Path,
) -> None:
    pd.DataFrame({"spotID": adata_st.obs_names, "cell_state": spot_states}).to_csv(
        output_path, index=False
    )
    logger.info("Spot mapping → %s", output_path)


def cell_state_fractions(cell_states: np.ndarray, K: int) -> dict[int, float]:
    counts = np.bincount(cell_states, minlength=K)
    return {k: float(counts[k]) / len(cell_states) for k in range(K)}


def plot_cell_state_pie(
    cell_fractions: dict[int, float],
    spot_fractions: dict[int, float],
    output_path: Path,
) -> None:
    K = len(cell_fractions)
    labels = [f"State {k}" for k in range(K)]
    cell_sizes = [cell_fractions[k] for k in range(K)]
    spot_sizes = [spot_fractions[k] for k in range(K)]

    fig, (ax_cell, ax_spot) = plt.subplots(1, 2, figsize=(12, 6))

    ax_cell.pie(cell_sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax_cell.set_title("Cell fraction per cell state")

    nonzero = [(l, s) for l, s in zip(labels, spot_sizes) if s > 0]
    spot_labels_nz, spot_sizes_nz = zip(*nonzero) if nonzero else ([], [])
    ax_spot.pie(spot_sizes_nz, labels=spot_labels_nz, autopct="%1.1f%%", startangle=90)
    ax_spot.set_title("Spot fraction per cell state")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Pie chart → %s", output_path)


def cell_state_anndata(
    adata_sc: AnnData,
    cell_states: np.ndarray,
    K: int,
) -> AnnData:
    """
    Returns AnnData of shape (K, G_sc) where X[k] is the mean expression
    of all cells assigned to state k.
    """
    X = _dense_X(adata_sc)
    means = np.zeros((K, X.shape[1]), dtype=np.float32)
    for k in range(K):
        mask = cell_states == k
        if mask.any():
            means[k] = X[mask].mean(axis=0)

    obs_df = pd.DataFrame(
        {
            "cell_state": np.arange(K),
            "n_cells": [int((cell_states == k).sum()) for k in range(K)],
        },
        index=[f"state_{k}" for k in range(K)],
    )
    return ad.AnnData(X=means, obs=obs_df, var=adata_sc.var.copy())
