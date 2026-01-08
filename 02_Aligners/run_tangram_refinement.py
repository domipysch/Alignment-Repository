import os
import tangram_refined.refined_tangram as tg
import pandas as pd
import scanpy as sc
import numpy as np
from utils import load_sc_adata, load_st_adata, fmt_nonzero_4
import argparse
import logging
from scipy.sparse import issparse


def tangram_refinement_align_data(
    dataset_folder: str,
    normalize_and_log: bool,
    deterministic_mapping: bool,
    compute_marker_genes: bool,
    map_clusters: bool,
    cell_type_key: str,
    output_path: str
):
    """
    Run refined Tangram alignment on a prepared dataset in the given folder.
    Saves predicted gene expression per spot (GEP) as CSV to output_path.

    Args:
        dataset_folder: Path to dataset folder
        normalize_and_log: Should the sc and st input data be normalized and log-transformed before alignment?
        deterministic_mapping: Should the cell-to-spot mapping be turned deterministic before multiplication wit sc-data?
            (one cell type per spot, one hot encoding)
        compute_marker_genes: Whether to compute marker genes (as proposed in Tangram Tutorials) or use all genes.
        map_clusters: Whether to use cluster-based mapping (cell types) or cell-based mapping.
        cell_type_key: What cell type key to load from sc data as cell type annotation.
        output_path: Full path where to save the resulting GEP CSV.

    Returns: None
    """
    assert os.path.isdir(dataset_folder), f"Dataset folder not found: {dataset_folder}"

    logger.info("Load data")
    adata_sc = load_sc_adata(dataset_folder, cell_type_key=cell_type_key)
    adata_st = load_st_adata(dataset_folder)
    logger.info("Data loaded")
    logger.info(f"Single Cell Data: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    logger.info(f"Spatial Data: {adata_st.n_obs} spots x {adata_st.n_vars} genes")

    # Step 0 (optional): Compute marker genes
    if compute_marker_genes:
        logger.info("Define marker genes")
        adata_sc_copy = adata_sc.copy()

        # Filter out cell types with only one cell for marker gene computation
        singletons = adata_sc_copy.obs['cell_subclass'].value_counts().loc[lambda x: x == 1].index.tolist()
        adata_sc_copy = adata_sc_copy[~adata_sc_copy.obs['cell_subclass'].isin(singletons)].copy()

        # Proposed in Tangram tutorials: normalize & log transform first
        sc.pp.normalize_total(adata_sc_copy)
        adata_sc_copy.X = np.log1p(adata_sc_copy.X)

        sc.tl.rank_genes_groups(adata_sc_copy, groupby="cell_subclass", use_raw=False)
        markers_df = pd.DataFrame(adata_sc_copy.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
        markers = list(np.unique(markers_df.melt().value.values))

    if normalize_and_log:
        # Normalize & log-transform gene expression and spatial data
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        adata_sc.X = np.log1p(adata_sc.X)
        adata_st.X = np.log1p(adata_st.X)

    # Step 1: Tangram pre-processing
    logger.info("Pre-process data with Tangram")
    if compute_marker_genes:
        tg.pp_adatas(adata_sc, adata_st, genes=markers)
    else:
        tg.pp_adatas(adata_sc, adata_st, genes=None)

    # Step 2: Mapping to space
    logger.info("Map cells to spots with Tangram")
    if map_clusters:
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_st,
            mode="clusters",
            cluster_label='cell_subclass',  # .obs field with cell types
            density_prior='rna_count_based',
            num_epochs=500,
            device='cpu',
            lambda_r=2.95e-09,
            lambda_l2=1e-18,
            lambda_neighborhood_g1=0.99,
            lambda_getis_ord=0.71,
            lambda_ct_islands=0.17
        )
    else:
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_st,
            mode="cells",
            density_prior='rna_count_based',
            num_epochs=500,
            device='cpu',
            lambda_r=2.95e-09,
            lambda_l2=1e-18,
            lambda_neighborhood_g1=0.99,
            lambda_getis_ord=0.71,
            lambda_ct_islands=0.17
        )

    # Step 3 (optional): Apply one-hot encoding to mapping
    if deterministic_mapping:
        if issparse(ad_map.X):
            mat = ad_map.X.toarray()
        else:
            mat = ad_map.X.copy()
        argmax_idx = np.argmax(mat, axis=1)
        one_hot = np.zeros_like(mat, dtype=float)
        one_hot[np.arange(mat.shape[0]), argmax_idx] = 1.0
        ad_map.X = one_hot

    # Step 4: Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
    ad_ge = tg.project_genes(
        adata_map=ad_map,
        adata_sc=adata_sc,
        cluster_label="cell_subclass" if map_clusters else None
    )

    # Export ad_ge to CSV
    # - Rows: Genes
    # - Columns: Spots
    # - Top-left cell = "GEP"
    if issparse(ad_ge.X):
        expr = ad_ge.X.A
    else:
        expr = ad_ge.X

    # Check: Rows = Genes, Columns = Spots
    assert expr.T.shape == (adata_sc.n_vars, adata_st.n_obs), "dims passen nicht"

    df = pd.DataFrame(expr.T, index=list(s.upper() for s in ad_ge.var_names), columns=ad_ge.obs_names)
    df_formatted = df.map(fmt_nonzero_4)
    df_formatted.to_csv(output_path, index=True, index_label="GEP")  # "GEP" in cell 0,0
    logger.info(f"Saved tangram GEP to {output_path}")

    # Step 4.2: Optional: which cell types are present in spots (left commented)
    # tg.project_cell_annotations(ad_map, adata_st, annotation="cell_subclass")
    # annotation_list = list(pd.unique(adata_sc.obs['cell_subclass']))
    # tg.plot_cell_annotation_sc(adata_st, annotation_list, perc=0.02)


if __name__ == "__main__":
    """
    Run Tangram Refinement alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run refined Tangram alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Path to dataset folder')
    args = parser.parse_args()

    tangram_refinement_align_data(
        args.dataset,
        normalize_and_log=False,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=False,
        cell_type_key="cellType",
        output_path=os.path.join(args.dataset, "results_cell", "tangram_refinement_non-det_GEP.csv")
    )
    tangram_refinement_align_data(
        args.dataset,
        normalize_and_log=False,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellType",
        output_path=os.path.join(args.dataset, "results_cellType", "tangram_refinement_non-det_GEP.csv")
    )
    tangram_refinement_align_data(
        args.dataset,
        normalize_and_log=False,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.dataset, "results_cellTypeMinor", "tangram_refinement_non-det_GEP.csv")
    )
