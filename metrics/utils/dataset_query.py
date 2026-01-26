from typing import List, Tuple
from pathlib import Path
import pandas as pd


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


def get_z_real_and_predicted_data(dataset_folder: Path, result_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read input ST data and predicted Z' data from result file, filter both to shared marker genes only.

    Args:
        dataset_folder: pathlib.Path zum Dataset-Ordner
        result_file: Path zur Ergebnisdatei

    Returns:
        Two DataFrames (ST data, Z' data).
    """

    # Check if file exists
    st_path = dataset_folder / "stData_GEP.csv"
    if not st_path.exists():
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {st_path}")

    # Check if result file exists
    if not result_file.exists():
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {result_file}")

    # ST data DataFrames einlesen
    df_st = pd.read_csv(st_path, header=0, index_col=0)

    # Z' DataFrame aus Ergebnisdatei einlesen
    df_res = pd.read_csv(result_file, header=0, index_col=0)

    # Filtern nach marker genes
    marker_genes = set(get_shared_genes(dataset_folder))
    df_st = df_st.loc[df_st.index.isin(marker_genes)]
    df_res = df_res.loc[df_res.index.isin(marker_genes)]

    # Reindex to same order
    df_res = df_res.reindex(index=df_st.index, columns=df_st.columns)
    assert df_st.index.equals(df_res.index), "Gene sind nicht in der gleichen Reihenfolge oder nicht identisch."
    assert df_st.columns.equals(df_res.columns), "Spots sind nicht in der gleichen Reihenfolge oder nicht identisch."

    return df_st, df_res
