import os
from pathlib import Path
import tacco as tc
import pandas as pd
import numpy as np
import logging
from anndata import AnnData
import argparse
from scipy.sparse import issparse
from ..utils.io import load_sc_adata, load_st_adata


def tacco_align_data(
    dataset_folder: str,
    deterministic_mapping: bool,
    map_cell_types: bool,
    cell_type_key: str = "cellType",
    output_path: Path = None,
) -> AnnData:
    """
    Run TACCO alignment on a prepared dataset in the given folder.
    Saves predicted gene expression per spot (GEP) as h5ad to output_path.

    Args:
        dataset_folder: Path to dataset folder.
        deterministic_mapping: Convert the probabilistic mapping to one-hot (one cell/type per spot).
        map_cell_types: If True, aggregate cells by cell_type_key before mapping.
                        If False, map individual cells directly.
        cell_type_key: obs column to use as annotation when map_cell_types=True.
        output_path: Full path where to save the resulting GEP h5ad.

    Returns:
        AnnData with obs=genes, var=spots (G x S layout).
    """
    assert os.path.isdir(dataset_folder), f"Dataset folder not found: {dataset_folder}"

    logging.info("Load data")
    adata_sc = load_sc_adata(Path(dataset_folder))  # C x G
    adata_st = load_st_adata(Path(dataset_folder))  # S x G

    # Determine which obs column to use as the annotation key for TACCO
    if map_cell_types:
        if cell_type_key not in adata_sc.obs.columns:
            raise KeyError(
                f"cell_type_key '{cell_type_key}' not found in obs columns "
                f"{list(adata_sc.obs.columns)}."
            )
        annotation_col = cell_type_key
    else:
        # Map individual cells: expose obs_names (cellID) as a column
        annotation_col = adata_sc.obs.index.name or "cellID"
        adata_sc.obs[annotation_col] = adata_sc.obs_names.tolist()

    # Change the datatype to float32 to avoid potential issues with TACCO
    adata_sc.X = adata_sc.X.astype(np.float32)
    adata_st.X = adata_st.X.astype(np.float32)

    # Map with TACCO
    logging.info("Align data with TACCO (annotation_col=%s)", annotation_col)
    tc.tl.annotate(
        adata_st,
        adata_sc,
        annotation_key=annotation_col,
        result_key="align_result",
    )
    # Mapping now in adata_st.obsm["align_result"]

    # Compute mean expression per annotation group (cells x genes -> groups x genes)
    if issparse(adata_sc.X):
        Xsc = adata_sc.X.toarray()
    else:
        Xsc = np.array(adata_sc.X)
    sc_obs = pd.DataFrame(
        Xsc, index=adata_sc.obs_names, columns=adata_sc.var_names
    )  # C x G
    mean_expr = sc_obs.groupby(adata_sc.obs[annotation_col]).mean()  # T x G

    # Convert mean -> gene profile p_tg (rows sum to 1)
    p = mean_expr.values
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum[p_sum == 0] = 1.0  # avoid division by zero for empty types
    p_tg = p / p_sum  # T x G
    logging.info("Shape p_tg: %s", p_tg.shape)

    # Fractions from TACCO (ensure row sums ~1), S x T
    fractions = pd.DataFrame(
        adata_st.obsm["align_result"], index=adata_st.obs_names, columns=mean_expr.index
    )
    logging.info("Shape fractions: %s", fractions)

    if deterministic_mapping:
        # For each row (spot) in fractions, set the max value to 1 and all others to 0 (map each spot to exactly one cell type)
        idxmax_series = fractions.idxmax(axis=1)
        # Map column labels to integer indices
        col_idx = fractions.columns.get_indexer(idxmax_series)
        # build one-hot values and wrap back to DataFrame
        one_hot_vals = np.zeros_like(fractions.values, dtype=float)
        one_hot_vals[np.arange(len(fractions)), col_idx] = 1.0
        one_hot = pd.DataFrame(
            one_hot_vals, index=fractions.index, columns=fractions.columns, dtype=float
        )
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

    # Build result AnnData (layout: obs = genes G, var = spots S)
    adata_recon = AnnData(
        X=recon.T.astype(np.float32),
        obs=pd.DataFrame(index=adata_sc.var_names),
        var=pd.DataFrame(index=adata_st.obs_names),
    )
    adata_recon.obs_names = list(adata_sc.var_names)
    adata_recon.var_names = list(adata_st.obs_names)

    # Write h5ad
    output_path = Path(output_path).with_suffix(".h5ad")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_recon.write_h5ad(output_path)
    logging.info("Saved tacco GEP to %s", output_path)

    return adata_recon


if __name__ == "__main__":
    """
    Run TACCO alignment on a prepared dataset at given folder.
    Settings can be modified in the code below.
    """

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run TACCO alignment on a dataset folder"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="Path to dataset folder (default: development workspace mouse cortex)",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Path where to store result"
    )
    parser.add_argument(
        "-det",
        "--deterministic",
        action="store_true",
        help="Whether to apply argmax to mapping",
    )
    parser.add_argument(
        "--map-cell-types",
        action="store_true",
        default=False,
        help="If set, aggregate cells by cell_type_key before mapping. Otherwise map individual cells.",
    )
    parser.add_argument(
        "--cell_type_key",
        type=str,
        required=False,
        default="cellType",
        help="obs column to use as cell type annotation (only used when --map-cell-types is set).",
    )
    args = parser.parse_args()

    tacco_align_data(
        args.dataset,
        deterministic_mapping=args.deterministic,
        map_cell_types=args.map_cell_types,
        cell_type_key=args.cell_type_key,
        output_path=args.output_path,
    )
