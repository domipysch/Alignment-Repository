import os
import pandas as pd
import anndata as ad


def load_sc_adata(dataset_folder: str, cell_type_key: str = "cellType") -> ad.AnnData:
    """
    Load single-cell data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
        cell_type_key: Key in scData_cells.csv file to read cell type annotations from
    Returns:
        ad.AnnData: Single-cell AnnData object (C x G)
    """
    # Cells = Rows, Genes = Columns
    df = pd.read_csv(os.path.join(dataset_folder, "scData_GEP.csv"), index_col=0)
    adata_sc = ad.AnnData(df.T)
    # Load cell types as annotation, if available
    cells = pd.read_csv(os.path.join(dataset_folder, "scData_Cells.csv"))
    if cell_type_key in cells.columns:
        adata_sc.obs["cell_subclass"] = cells[cell_type_key].values
    return adata_sc


def load_st_adata(dataset_folder: str) -> ad.AnnData:
    """
    Load ST data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
    Returns:
        ad.AnnData: ST AnnData object (S x G)
    """
    # Spots = Rows, Genes = Columns
    df = pd.read_csv(os.path.join(dataset_folder, "stData_GEP.csv"), index_col=0)
    adata_st = ad.AnnData(df.T)
    # Load spot coordinates
    coords = pd.read_csv(os.path.join(dataset_folder, "stData_Spots.csv"), index_col=0)
    adata_st.obsm["spatial"] = coords[["cArray0", "cArray1"]].values
    return adata_st


def fmt_nonzero_4(x: float) -> str:
    """
    Format a numeric value for display to cap at up to four decimal places.

    Args:
        x: Input value (float)
    Returns:
        str: Formatted string
    """
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
    except Exception:
        raise Exception("Input value is not convertible to float")
    if xf == 0.0:
        return "0.0"
    return f"{xf:.4f}"