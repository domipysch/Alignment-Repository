import os
from typing import Dict
from .spatial_graph import SpatialGraphType
import anndata as ad
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def load_sc_adata(dataset_folder: str) -> ad.AnnData:
    """
    Load single-cell data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
    Returns:
        ad.AnnData: Single-cell AnnData object (C x G)
    """
    logger.debug("Load scRNA data")
    # In file: Genes = Rows, Cells = Columns
    df = pd.read_csv(os.path.join(dataset_folder, "scData_GEP.csv"), index_col=0)
    adata_sc = ad.AnnData(df.T)
    return adata_sc


def load_st_adata(dataset_folder: str) -> ad.AnnData:
    """
    Load ST data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
    Returns:
        ad.AnnData: ST AnnData object (S x G)
    """
    logger.debug("Load ST data")
    # In file: Genes = Rows, Spots = Columns
    df = pd.read_csv(os.path.join(dataset_folder, "stData_GEP.csv"), index_col=0)
    adata_st = ad.AnnData(df.T)
    # Load spot coordinates
    logger.debug("Load ST coordinates")
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


def graph_type_from_config(graph_cfg: Dict) -> SpatialGraphType:

    if not isinstance(graph_cfg, dict):
        raise ValueError("`graph_cfg` must be a dict.")

    graph_type = graph_cfg.get("type")
    if not isinstance(graph_type, str):
        raise ValueError("`graph.type` must be a string in the config.")

    t = graph_type.strip().lower()

    if t == "knn":
        return SpatialGraphType.KNN
    if t == "mutual_knn":
        return SpatialGraphType.MUTUAL_KNN
    if t == "radius":
        return SpatialGraphType.RADIUS
    if t == "delaunay":
        return SpatialGraphType.DELAUNAY

    raise ValueError(f"Unsupported graph.type: '{graph_type}'. Expected one of: knn, mutual_knn, radius, delaunay.")
