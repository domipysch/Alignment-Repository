from typing import List, Tuple
from pathlib import Path
import pandas as pd
from anndata import AnnData
from ...utils.io import csv_to_anndata


def get_sc_genes(dataset_folder: Path) -> List[str]:
    """
    Get the list of all genes in the scRNA data of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder containing scData_Genes.csv.

    Returns:
        List of gene ID strings in file order.
    """
    csv_path = dataset_folder / "scData_Genes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"scData_Genes.csv not found at: {csv_path}")

    genes: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.lower().startswith("geneid"):
                continue
            if "," in line:
                first = line.split(",", 1)[0]
            else:
                first = line
            genes.append(first)
    return genes


def get_st_genes(dataset_folder: Path) -> List[str]:
    """
    Get the list of all genes in the ST data of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder containing stData_Genes.csv.

    Returns:
        List of gene ID strings in file order.
    """
    csv_path = dataset_folder / "stData_Genes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"stData_Genes.csv not found at: {csv_path}")

    genes: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip header
            if i == 0 and line.lower().startswith("geneid"):
                continue
            if "," in line:
                first = line.split(",", 1)[0]
            else:
                first = line
            genes.append(first)
    return genes


def get_shared_genes(dataset_folder: Path) -> List[str]:
    """
    Get the list of genes shared between scRNA and ST data of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder.

    Returns:
        List of gene ID strings present in both scRNA and ST data (order not guaranteed).
    """
    sc_genes = set(get_sc_genes(dataset_folder))
    st_genes = set(get_st_genes(dataset_folder))
    shared_genes = list(sc_genes.intersection(st_genes))
    return shared_genes


def get_cell_annotations(dataset_folder: Path) -> pd.DataFrame:
    """
    Load cell annotations from scData_Annotations.csv of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder containing scData_Annotations.csv.

    Returns:
        DataFrame of cell annotations with cell IDs as index.
    """
    csv_path = dataset_folder / "scData_Annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"scData_Annotations.csv not found at: {csv_path}")

    df_annotations = pd.read_csv(csv_path, header=0, index_col=0)
    return df_annotations


def get_z_real_and_predicted_data_only_shared_genes(
    dataset_folder: Path, result_gep: AnnData
) -> Tuple[AnnData, AnnData]:
    """
    Load input ST data and predicted Z', filtered to shared marker genes only.

    Args:
        dataset_folder: Path to the dataset folder containing stData_GEP.csv.
        result_gep: Predicted gene expression as AnnData (G x S).

    Returns:
        Tuple (st_shared, result_shared): both AnnData objects of shape (S x shared_G),
        aligned to the same set of shared genes in the same order.
    """
    st_path = dataset_folder / "stData_GEP.csv"
    if not st_path.exists():
        raise FileNotFoundError(f"ST data file not found: {st_path}")

    st_ad = csv_to_anndata(st_path, transpose=True)
    result_gep = result_gep.transpose()

    # Filter to shared marker genes, preserving the order from st_ad
    marker_genes = set(get_shared_genes(dataset_folder))
    common_genes = [
        g for g in st_ad.var_names if g in marker_genes and g in result_gep.var_names
    ]
    st_shared_ad = st_ad[:, common_genes].copy()
    result_shared_gep = result_gep[:, common_genes].copy()

    assert st_shared_ad.var_names.equals(
        result_shared_gep.var_names
    ), "Gene order or identity mismatch between ST data and result GEP."

    return st_shared_ad, result_shared_gep
