import os
import tangram as tg
import pandas as pd
import scanpy as sc
import numpy as np
import logging
from utils import load_sc_adata, load_st_adata, fmt_nonzero_4
import argparse
from scipy.sparse import issparse


def tangram_align_data(
    dataset_folder: str,
    normalize_and_log: bool,
    deterministic_mapping: bool,
    compute_marker_genes: bool,
    map_clusters: bool,
    cell_type_key: str,
    output_path: str,
):
    """
    Run Tangram alignment on a prepared dataset in given folder.
    Saves prediction GEP CSV to output_path.

    Args:
        dataset_folder: Path to dataset folder
        normalize_and_log: Should the sc and st input data be normalized and log-transformed before alignment?
        deterministic_mapping: Should the cell-to-spot mapping be turned deterministic before multiplication wit sc-data?
            (one cell type per spot, one hot encoding)
        compute_marker_genes: Whether to compute marker genes (as proposed in Tangram Tutorials) or use all genes.
        map_clusters: Whether to use cluster-based mapping (cell types) or cell-based mapping.
        cell_type_key: What cell type key to load from sc data as cell type annotation.
        output_path: Full path where to save the resulting GEP CSV.

    Returns: -
    """
    assert os.path.isdir(dataset_folder), f"Dataset folder not found: {dataset_folder}"

    # Load input scRNA and ST data
    logger.info("Load data")
    adata_sc = load_sc_adata(dataset_folder, cell_type_key=cell_type_key)  # C x G
    adata_st = load_st_adata(dataset_folder)  # S x G
    logger.info("Data loaded")
    logger.info(f"Single Cell Data: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    logger.info(f"Spatial Data: {adata_st.n_obs} spots x {adata_st.n_vars} genes")

    # Step 1 (optional): Compute marker genes (optional, speeds up mapping)
    if compute_marker_genes:
        logger.info("Define marker genes")
        adata_sc_copy = adata_sc.copy()

        # Filter out cell types with only one cell for marker gene computation
        singletons = adata_sc_copy.obs['cell_subclass'].value_counts().loc[lambda x: x == 1].index.tolist()
        adata_sc_copy = adata_sc_copy[~adata_sc_copy.obs['cell_subclass'].isin(singletons)].copy()

        # Proposed in Tangram tutorials: normalize & log transform first
        sc.pp.normalize_total(adata_sc_copy)
        adata_sc_copy.X = np.log1p(adata_sc_copy.X)

        # Create list of names of marker genes
        sc.tl.rank_genes_groups(adata_sc_copy, groupby="cell_subclass", use_raw=False)
        markers_df = pd.DataFrame(adata_sc_copy.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
        markers: list[str] = list(np.unique(markers_df.melt().value.values))

    if normalize_and_log:
        # Normalize & log-transform input data (optional, as in Tangram tutorials)
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        adata_sc.X = np.log1p(adata_sc.X)
        adata_st.X = np.log1p(adata_st.X)

    # Step 2: Tangram pre-processing
    # (see https://github.com/broadinstitute/Tangram/blob/master/tangram/mapping_utils.py)
    if compute_marker_genes:
        logger.info(f"Pre-process data with Tangram with {len(markers)} marker genes")
        tg.pp_adatas(adata_sc, adata_st, genes=markers)
    else:
        logger.info(f"Pre-process data with Tangram with all genes as marker genes")
        tg.pp_adatas(adata_sc, adata_st, genes=None)

    # Step 3: Mapping
    logger.info("Map cells to spots with Tangram")
    if map_clusters:
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_st,
            mode="clusters",
            cluster_label='cell_subclass', # .obs field w cell types
            density_prior='rna_count_based',
            num_epochs=500,
            device='cpu',
        )  # T x S
        assert ad_map.n_obs == len(adata_sc.obs['cell_subclass'].unique())
        assert ad_map.n_vars == adata_st.n_obs
    else:
        ad_map = tg.map_cells_to_space(
            adata_sc,
            adata_st,
            mode="cells",
            density_prior='rna_count_based',
            num_epochs=500,
            device='cpu',
        )  # C x S
        assert ad_map.n_obs == adata_sc.n_obs
        assert ad_map.n_vars == adata_st.n_obs

    # Step 4 (optional): Apply one-hot encoding to mapping: Only one cell / cell type per spot
    logger.info("Apply deterministic mapping" if deterministic_mapping else "Keep probabilistic mapping")
    if deterministic_mapping:
        # For each column (spot) in ad_map, set the max value to 1 and all others to 0 (map spot to exactly one cell)
        if issparse(ad_map.X):
            mat = ad_map.X.toarray()
        else:
            mat = ad_map.X.copy()
        argmax_idx = np.argmax(mat, axis=0)  # for each spot (column), index of max cell / cell type
        one_hot = np.zeros_like(mat, dtype=float)
        one_hot[np.arange(mat.shape[0]), argmax_idx] = 1.0
        ad_map.X = one_hot

    # Step 5: Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
    logger.info("Project gene expression to spatial spots")
    ad_ge = tg.project_genes(
        adata_map=ad_map,
        adata_sc=adata_sc,
        cluster_label="cell_subclass" if map_clusters else None
    )  # S x G

    # Transpose to G x S
    ad_ge = ad_ge.transpose()  # now G x S
    assert ad_ge.n_obs == adata_sc.n_vars
    assert ad_ge.n_vars == adata_st.n_obs

    # Export ad_ge to CSV
    # - Rows: Genes
    # - Columns: Spots
    # - Top left cell = "GEP"
    if issparse(ad_ge.X):
        expr = ad_ge.X.A
    else:
        expr = ad_ge.X

    # Check: Rows = Genes, Columns = Spots
    assert expr.shape == (adata_sc.n_vars, adata_st.n_obs), "dims passen nicht"

    # Step 6: Write CSV
    df = pd.DataFrame(expr, index=list(s.upper() for s in ad_ge.obs_names), columns=ad_ge.var_names)
    df_formatted = df.map(fmt_nonzero_4)
    df_formatted.to_csv(output_path, index=True, index_label="GEP")  # "GEP" in cell 0,0
    logger.info(f"Saved tangram GEP to {output_path}")


if __name__ == "__main__":
    """
    Run Tangram alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run Tangram alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Path to dataset folder')
    args = parser.parse_args()

    tangram_align_data(
        args.dataset,
        normalize_and_log=False,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=False,
        cell_type_key="cellType",
        output_path=os.path.join(args.dataset, "results_cell", "tangram_GEP.csv")
    )
    tangram_align_data(
        args.dataset,
        normalize_and_log=False,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellType",
        output_path=os.path.join(args.dataset, "results_cellType", "tangram_GEP.csv")
    )
    tangram_align_data(
        args.dataset,
        normalize_and_log=False,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.dataset, "results_cellTypeMinor", "tangram_GEP.csv")
    )
