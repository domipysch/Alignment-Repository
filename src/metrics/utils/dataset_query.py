from typing import List, Tuple
from pathlib import Path
import pandas as pd
from anndata import AnnData
from ...utils.io import load_sc_adata, load_st_adata


def get_sc_genes(dataset_folder: Path) -> List[str]:
    """
    Get the list of all genes in the scRNA data of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder containing sc.h5ad.

    Returns:
        List of gene ID strings in file order.
    """
    return load_sc_adata(dataset_folder).var_names.tolist()


def get_st_genes(dataset_folder: Path) -> List[str]:
    """
    Get the list of all genes in the ST data of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder containing st.h5ad.

    Returns:
        List of gene ID strings in file order.
    """
    return load_st_adata(dataset_folder).var_names.tolist()


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
    Load cell annotations from sc.h5ad of a given dataset.

    Args:
        dataset_folder: Path to the dataset folder containing sc.h5ad.

    Returns:
        DataFrame of cell annotations with cell IDs as index.
    """
    return load_sc_adata(dataset_folder).obs


def get_z_real_and_predicted_data_only_shared_genes(
    dataset_folder: Path, result_gep: AnnData
) -> Tuple[AnnData, AnnData]:
    """
    Load input ST data and predicted Z', filtered to shared marker genes only.

    Args:
        dataset_folder: Path to the dataset folder containing st.h5ad.
        result_gep: Predicted gene expression as AnnData (G x S).

    Returns:
        Tuple (st_shared, result_shared): both AnnData objects of shape (S x shared_G),
        aligned to the same set of shared genes in the same order.
    """
    st_ad = load_st_adata(dataset_folder)
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
