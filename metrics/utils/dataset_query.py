from typing import List, Tuple
from pathlib import Path
import pandas as pd
from anndata import AnnData
from ...utils.io import csv_to_anndata


def get_sc_genes(dataset_folder: Path) -> List[str]:
    """
    Get a list of all genes contained in scRNA data of given dataset
    """

    csv_path = dataset_folder / "scData_Genes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"stData_Genes.csv nicht gefunden unter: {csv_path}")

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
    Get a list of all genes contained in ST data of given dataset
    """
    csv_path = dataset_folder / "stData_Genes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"stData_Genes.csv nicht gefunden unter: {csv_path}")

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
    Get a list of genes shared between scRNA and ST data of given dataset
    """
    sc_genes = set(get_sc_genes(dataset_folder))
    st_genes = set(get_st_genes(dataset_folder))
    shared_genes = list(sc_genes.intersection(st_genes))
    return shared_genes


def get_cell_annotations(dataset_folder: Path) -> pd.DataFrame:
    """
    Load 'scData_Cells.csv' of given dataset, return as DataFrame
    """
    csv_path = dataset_folder / "scData_Annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"scData_Annotations.csv nicht gefunden unter: {csv_path}")

    df_annotations = pd.read_csv(csv_path, header=0, index_col=0)
    return df_annotations


def get_z_real_and_predicted_data_only_shared_genes(dataset_folder: Path, result_gep: AnnData) -> Tuple[AnnData, AnnData]:
    """
    Read input ST data and predicted Z' data from result file, filter both to shared marker genes only.

    Args:
        dataset_folder: pathlib.Path zum Dataset-Ordner
        result_gep: G x S

    Returns:
        Two anndata (ST data, Z' data), S x shared G.
    """

    # Check if file exists
    st_path = dataset_folder / "stData_GEP.csv"
    if not st_path.exists():
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {st_path}")

    # ST data DataFrames einlesen
    st_ad = csv_to_anndata(st_path, transpose=True)
    result_gep = result_gep.transpose()

    # Filtern nach marker genes
    marker_genes = set(get_shared_genes(dataset_folder))

    # Bestimme gemeinsame Gene in der Reihenfolge von st_ad
    common_genes = [g for g in st_ad.var_names if g in marker_genes and g in result_gep.var_names]
    st_shared_ad = st_ad[:, common_genes].copy()
    result_shared_gep = result_gep[:, common_genes].copy()

    # Sicherstellen, dass Reihenfolge und Identität übereinstimmen
    assert st_shared_ad.var_names.equals(result_shared_gep.var_names), "Gene sind nicht in der gleichen Reihenfolge oder nicht identisch."

    return st_shared_ad, result_shared_gep

