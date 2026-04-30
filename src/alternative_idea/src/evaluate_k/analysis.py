"""
Post-mapping analysis for AlternativeIdea.

Standalone entry point; call run_analysis() after mapping:

    from src.alternative_idea.src.evaluate_k.analysis import run_analysis
    results = run_analysis(adata_sc, adata_st, B, C, output_dir=Path("analysis_out"), K=5)

Outputs written to output_dir:
    data/cell_mapping.csv            cellID → hard cell-state assignment
    data/spot_mapping.csv            spotID → hard cell-state assignment
    data/cell_state_profiles.h5ad    AnnData (K × G_sc): mean expression per state
    cell_state_profiles.png          per-state expression heatmap + cell/spot fractions
    cell_state_fractions.png         standalone cell/spot fraction bar charts
    gep_distance_comparison.png      pairwise cosine-distance heatmaps: computed states, Leiden shared, Leiden all
    [auc_matching.png disabled — code preserved in matching.py, re-enable via compute_auc_matching]
    umap_comparison.png              UMAP: computed assignment vs Leiden (shared & all genes)
    unsupervised_metrics.csv         silhouette / Dunn / modularity / centroid cosim
    crosstab_heatmap.png             predicted state × GT cell type (only if gt_label_key given)
    supervised_metrics.csv           Hungarian, greedy, contingency, AUC scores
    centroid_matching_hungarian.png  Leiden-vs-computed, Hungarian matching
    centroid_matching_greedy.png     Leiden-vs-computed, greedy matching
    contingency_heatmap.png          contingency matrix (Leiden, all genes)
    auc_matching.png                 AUC matching heatmap (Leiden, all genes)

Assumptions
-----------
- adata_sc.X holds raw count data (analysis internally normalises a working copy).
  If your adata is already log-normalised, pass normalize=False to run_analysis.
- B and C may be torch.Tensor or np.ndarray on any device.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData

from ._utils import _to_numpy, _prepare_for_umap, hard_assignments
from .assignments import (
    save_cell_mapping_csv,
    save_spot_mapping_csv,
    cell_state_fractions,
    cell_state_anndata,
)
from .clustering import compute_all_metrics, run_leiden_shared_genes
from .matching import (
    compute_leiden_vs_computed_matching,
    plot_centroid_matching_heatmap,
    plot_centroid_matching_greedy,
    compute_contingency_matching,
    plot_contingency_heatmap,
)
from .plots import (
    plot_umap_comparison,
    plot_crosstab_heatmap,
    plot_state_profiles,
    plot_state_fractions,
    plot_gep_distance_comparison,
)

logger = logging.getLogger(__name__)


def run_analysis(
    adata_sc: AnnData,
    adata_st: AnnData,
    B: torch.Tensor | np.ndarray,
    C: torch.Tensor | np.ndarray,
    output_dir: Path,
    K: int | None = None,
    leiden_resolution: float = 0.5,
    leiden_resolution_fine: float = 2.0,  # kept for API compatibility, no longer used
    normalize: bool = True,
    gt_label_key: str | None = None,
    max_auc_iter: int = 50,  # kept for API compatibility, AUC matching disabled
) -> dict:
    """
    Full post-mapping analysis pipeline.

    Parameters
    ----------
    adata_sc         : Single-cell AnnData (cells × genes), raw counts expected.
    adata_st         : Spatial AnnData (spots × genes).
    B                : Cell-to-state soft-assignment matrix (n_cells, K).
    C                : Spot-to-state soft-assignment matrix (n_spots, K).
    output_dir       : Directory where all outputs are written (created if absent).
    K                : Number of cell states (inferred from B if None).
    leiden_resolution     : Resolution for the main Leiden reference clusterings.
    leiden_resolution_fine: Unused (kept for backwards-compatible call sites).
    max_auc_iter         : Maximum refinement iterations for AUC matching.
    normalize        : If True, normalize_total + log1p the working copy before
                       UMAP / metrics.  Set False if adata_sc is already
                       log-normalised.
    gt_label_key     : obs column name with ground-truth cell-type labels (e.g.
                       "cellType").  When provided, a crosstab heatmap of
                       predicted state × GT label is saved.  Silently skipped if
                       the column is absent from adata_sc.obs.

    Returns
    -------
    dict with keys:
        cell_states            np.ndarray (n_cells,)   hard cell-state labels
        spot_states            np.ndarray (n_spots,)   hard cell-state labels
        cell_fractions         dict[int, float]         fraction of cells per state
        spot_fractions         dict[int, float]         fraction of spots per state
        state_anndata          AnnData (K × G_sc)       mean expression per state
        metrics_computed       dict[str, float]         metrics for computed assignment
        metrics_leiden         dict[str, float]         metrics for Leiden (all genes)
        metrics_leiden_shared  dict[str, float]         metrics for Leiden (shared genes)
        centroid_matching    dict   Leiden(all genes)-vs-computed centroid matching
        contingency_matching dict   contingency argmax matching (Leiden, all genes)
        auc_matching         dict   AUC-based matching (fine Leiden, all genes)
        adata_processed        AnnData           sc data with UMAP + all labels
        adata_shared           AnnData           shared-gene subset with own UMAP
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B = _to_numpy(B)
    C = _to_numpy(C)
    if K is None:
        K = B.shape[1]

    shared_genes = list(set(adata_sc.var_names) & set(adata_st.var_names))
    logger.info(
        "run_analysis: K=%d, cells=%d, spots=%d, shared_genes=%d",
        K,
        len(adata_sc),
        len(adata_st),
        len(shared_genes),
    )

    # ── Hard assignments ──────────────────────────────────────────────────────
    cell_states = hard_assignments(B)
    spot_states = hard_assignments(C)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV outputs ───────────────────────────────────────────────────────────
    save_cell_mapping_csv(adata_sc, cell_states, data_dir / "cell_mapping.csv")
    save_spot_mapping_csv(adata_st, spot_states, data_dir / "spot_mapping.csv")

    # ── Fractions ─────────────────────────────────────────────────────────────
    cell_fractions = cell_state_fractions(cell_states, K)
    spot_fractions = cell_state_fractions(spot_states, K)

    # ── Cell-state AnnData ────────────────────────────────────────────────────
    adata_states = cell_state_anndata(adata_sc, cell_states, K)
    state_h5ad = data_dir / "cell_state_profiles.h5ad"
    adata_states.write_h5ad(state_h5ad)
    logger.info("Cell-state profiles → %s", state_h5ad)

    # ── Prepare sc data once (normalise, PCA, neighbors, UMAP) ───────────────
    adata_processed = _prepare_for_umap(adata_sc, normalize=normalize)

    # ── Computed clustering metrics ───────────────────────────────────────────
    logger.info("Computing metrics for the computed clustering…")
    metrics_computed = compute_all_metrics(adata_processed, cell_states)
    logger.info("Computed:  %s", metrics_computed)
    adata_processed.obs["computed_state"] = pd.Categorical(cell_states.astype(str))

    # ── Leiden reference – all genes ─────────────────────────────────────────
    logger.info(
        "Running Leiden clustering – all genes (resolution=%.2f)…", leiden_resolution
    )
    sc.tl.leiden(adata_processed, resolution=leiden_resolution, key_added="_leiden_ref")
    leiden_labels = adata_processed.obs["_leiden_ref"].astype(int).values
    logger.info("Leiden (all genes): %d clusters", len(np.unique(leiden_labels)))

    metrics_leiden = compute_all_metrics(adata_processed, leiden_labels)
    logger.info("Leiden (all genes): %s", metrics_leiden)
    adata_processed.obs["leiden_state"] = pd.Categorical(leiden_labels.astype(str))

    # ── Leiden reference – shared genes ───────────────────────────────────────
    logger.info(
        "Running Leiden clustering – shared genes (resolution=%.2f)…", leiden_resolution
    )
    leiden_shared_labels, adata_shared = run_leiden_shared_genes(
        adata_processed,
        shared_genes=shared_genes,
        resolution=leiden_resolution,
    )
    metrics_leiden_shared = compute_all_metrics(adata_shared, leiden_shared_labels)
    logger.info("Leiden (shared genes): %s", metrics_leiden_shared)

    adata_processed.obs["leiden_shared_state"] = pd.Categorical(
        leiden_shared_labels.astype(str)
    )

    # ── GEP centroid distance comparison ─────────────────────────────────────
    logger.info("Plotting GEP pairwise cosine-distance comparison…")
    plot_gep_distance_comparison(
        adata_norm=adata_processed,
        cell_states=cell_states,
        leiden_labels_all=leiden_labels,
        leiden_labels_shared=leiden_shared_labels,
        shared_genes=shared_genes,
        unique_computed=[int(x) for x in sorted(np.unique(cell_states))],
        unique_leiden_all=[int(x) for x in sorted(np.unique(leiden_labels))],
        unique_leiden_shared=[int(x) for x in sorted(np.unique(leiden_shared_labels))],
        output_path=output_dir / "gep_distance_comparison.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )

    # ── Combined UMAP comparison ──────────────────────────────────────────────
    plot_umap_comparison(
        adata_processed,
        panels=[
            ("computed_state", "Computed cell-state assignment"),
            (
                "leiden_shared_state",
                f"Leiden – shared genes (resolution={leiden_resolution})",
            ),
            ("leiden_state", f"Leiden – all genes (resolution={leiden_resolution})"),
        ],
        output_path=output_dir / "umap_comparison.png",
    )

    # ── Crosstab heatmap ──────────────────────────────────────────────────────
    if gt_label_key is not None:
        if gt_label_key in adata_sc.obs.columns:
            plot_crosstab_heatmap(
                cell_states,
                adata_sc.obs[gt_label_key].values,
                output_dir / "crosstab_heatmap.png",
                gt_label_name=gt_label_key,
            )
        else:
            logger.warning(
                "gt_label_key=%r not found in adata_sc.obs — skipping crosstab.",
                gt_label_key,
            )

    # ── Cell-state profiles ───────────────────────────────────────────────────
    logger.info("Plotting cell-state profiles…")
    plot_state_profiles(
        adata_sc,
        cell_states,
        shared_genes,
        output_dir / "cell_state_profiles.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )
    plot_state_fractions(
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
        unique_states=sorted(np.unique(cell_states).tolist()),
        output_path=output_dir / "cell_state_fractions.png",
    )

    # ── Centroid matching (all genes) ─────────────────────────────────────────
    logger.info("Computing Leiden-vs-computed centroid matching…")
    all_sc_genes = adata_sc.var_names.tolist()
    centroid_matching = compute_leiden_vs_computed_matching(
        adata_sc, cell_states, leiden_labels, all_sc_genes
    )
    plot_centroid_matching_heatmap(
        centroid_matching,
        output_dir / "centroid_matching_hungarian.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )
    plot_centroid_matching_greedy(
        centroid_matching,
        output_dir / "centroid_matching_greedy.png",
        cell_fractions=cell_fractions,
        spot_fractions=spot_fractions,
    )

    # ── Contingency matching (Leiden, all genes) ─────────────────────────────
    logger.info("Computing contingency matching…")
    contingency_matching = compute_contingency_matching(cell_states, leiden_labels)
    plot_contingency_heatmap(
        contingency_matching,
        output_dir / "contingency_heatmap.png",
        spot_fractions=spot_fractions,
    )

    # ── Supervised metrics CSV ────────────────────────────────────────────────
    pd.DataFrame(
        {
            "metric": [
                "hungarian_cosim",
                "greedy_cosim",
                "contingency_score",
            ],
            "value": [
                centroid_matching["hungarian_score"],
                centroid_matching["greedy_score"],
                contingency_matching["score"],
            ],
        }
    ).to_csv(output_dir / "supervised_metrics.csv", index=False)
    logger.info(
        "Supervised metrics → %s/supervised_metrics.csv  "
        "(Hungarian=%.3f | Greedy=%.3f | Contingency=%.3f)",
        output_dir,
        centroid_matching["hungarian_score"],
        centroid_matching["greedy_score"],
        contingency_matching["score"],
    )

    # ── Unsupervised metrics CSV ──────────────────────────────────────────────
    metrics_df = pd.DataFrame(
        {
            "metric": list(metrics_computed.keys()),
            "computed_assignment": list(metrics_computed.values()),
            "leiden_all_genes": [
                metrics_leiden.get(k, float("nan")) for k in metrics_computed
            ],
            "leiden_shared_genes": [
                metrics_leiden_shared.get(k, float("nan")) for k in metrics_computed
            ],
        }
    )
    metrics_df.to_csv(output_dir / "unsupervised_metrics.csv", index=False)
    logger.info("Unsupervised metrics → %s", output_dir / "unsupervised_metrics.csv")

    return {
        "cell_states": cell_states,
        "spot_states": spot_states,
        "cell_fractions": cell_fractions,
        "spot_fractions": spot_fractions,
        "state_anndata": adata_states,
        "metrics_computed": metrics_computed,
        "metrics_leiden": metrics_leiden,
        "metrics_leiden_shared": metrics_leiden_shared,
        "centroid_matching": centroid_matching,
        "contingency_matching": contingency_matching,
        "adata_processed": adata_processed,
        "adata_shared": adata_shared,
    }
