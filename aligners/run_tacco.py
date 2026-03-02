import os
import tacco as tc
import pandas as pd
import numpy as np
import logging

from anndata import AnnData

from utils import load_sc_adata, load_st_adata, fmt_nonzero_4
import argparse
from scipy.sparse import issparse


def tacco_align_data(
    dataset_folder: str,
    deterministic_mapping: bool,
    cell_type_key: str,
    output_path: str,
) -> AnnData:
    """
    Run TACCO alignment on a prepared dataset in the given folder.
    Saves predicted gene expression per spot (GEP) as CSV to output_path.

    Args:
        dataset_folder: Path to dataset folder
        deterministic_mapping: Should the cell-to-spot mapping be turned deterministic before multiplication wit sc-data?
            (one cell type per spot, one hot encoding)
        cell_type_key: What cell type key to load from sc data as cell type annotation.
        output_path: Full path where to save the resulting GEP CSV.

    Returns: None
    """
    assert os.path.isdir(dataset_folder), f"Dataset folder not found: {dataset_folder}"

    logging.info("Load data")
    adata_sc = load_sc_adata(dataset_folder, cell_type_key=cell_type_key)  # C x G
    adata_st = load_st_adata(dataset_folder)  # S x G

    # Change the datatype to float32 to avoid potential issues with TACCO
    adata_sc.X = adata_sc.X.astype(np.float32)
    adata_st.X = adata_st.X.astype(np.float32)

    # Map with TACCO
    logging.info("Align data with TACCO")
    tc.tl.annotate(
        adata_st,
        adata_sc,
        annotation_key="cell_subclass",
        result_key="align_result",
    )
    # Mapping now in adata_st.obsm["align_result"]

    # Compute mean per cell type (cells x genes -> types x genes)
    if issparse(adata_sc.X):
        Xsc = adata_sc.X.toarray()
    else:
        Xsc = np.array(adata_sc.X)
    sc_obs = pd.DataFrame(Xsc, index=adata_sc.obs_names, columns=adata_sc.var_names)  # C x G
    mean_expr = sc_obs.groupby(adata_sc.obs["cell_subclass"]).mean()  # T x G

    # Convert mean -> gene profile p_tg (rows sum to 1)
    p = mean_expr.values
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum[p_sum == 0] = 1.0  # avoid division by zero for empty types
    p_tg = p / p_sum  # T x G
    logging.info("Shape p_tg: %s", p_tg.shape)

    # Fractions from TACCO (ensure row sums ~1), S x T
    fractions = pd.DataFrame(adata_st.obsm["align_result"], index=adata_st.obs_names, columns=mean_expr.index)
    logging.info("Shape fractions: %s", fractions)

    if deterministic_mapping:
        # For each row (spot) in fractions, set the max value to 1 and all others to 0 (map each spot to exactly one cell type)
        idxmax_series = fractions.idxmax(axis=1)
        # Map column labels to integer indices
        col_idx = fractions.columns.get_indexer(idxmax_series)
        # build one-hot values and wrap back to DataFrame
        one_hot_vals = np.zeros_like(fractions.values, dtype=float)
        one_hot_vals[np.arange(len(fractions)), col_idx] = 1.0
        one_hot = pd.DataFrame(one_hot_vals, index=fractions.index, columns=fractions.columns, dtype=float)
        # replace fractions with the deterministic one-hot mapping
        fractions = one_hot  # S x T

    # Total counts per spot
    if issparse(adata_st.X):
        spot_counts = np.array(adata_st.X.sum(axis=1)).flatten()
    else:
        spot_counts = np.array(adata_st.X.sum(axis=1)).flatten()
    # shape: (spots,): per spot; sum
    C_st = fractions.to_numpy() * spot_counts[:, None]  # S x T

    # 6) reconstruct spot x gene
    recon = C_st @ p_tg  # S x G
    assert recon.shape == (adata_st.n_obs, adata_sc.n_vars), "dims passen nicht"

    # Export result to CSV
    # - Rows: Genes
    # - Columns: Spots
    # - Top left cell = "GEP"
    recon_df = pd.DataFrame(recon.T.astype(np.float32), index=adata_sc.var_names, columns=adata_st.obs_names)
    logging.info("Shape recon_df: %s", recon_df.shape)
    df_formatted = recon_df.map(fmt_nonzero_4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_formatted.to_csv(output_path, index=True, index_label="GEP")
    logging.info("Saved tacco GEP to %s", output_path)

    # Return reconstruction as AnnData (genes x spots) for potential further use
    adata_recon = AnnData(X=recon_df.values, obs=adata_sc.var, var=adata_st.obs)
    return adata_recon


if __name__ == "__main__":
    """
    Run TACCO alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run TACCO alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        help='Path to dataset folder (default: development workspace mouse cortex)')
    args = parser.parse_args()

    tacco_align_data(
        args.dataset,
        deterministic_mapping=True,
        cell_type_key="cellID",
        output_path=os.path.join(args.dataset, "results_cell", "tacco_GEP.csv")
    )
    tacco_align_data(
        args.dataset,
        deterministic_mapping=True,
        cell_type_key="cellType",
        output_path=os.path.join(args.dataset, "results_cellType", "tacco_GEP.csv")
    )
    tacco_align_data(
        args.dataset,
        deterministic_mapping=True,
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.dataset, "results_cellTypeMinor", "tacco_GEP.csv")
    )
