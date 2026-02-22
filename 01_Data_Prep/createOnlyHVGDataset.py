import os
import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import issparse
import scanpy as sc


def load_sc_adata(dataset_folder: str, cell_type_keys: list[str]) -> ad.AnnData:
    """
    Load single-cell data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
    Returns:
        ad.AnnData: Single-cell AnnData object (C x G)
    """
    # Cells = Rows, Genes = Columns
    df = pd.read_csv(os.path.join(dataset_folder, "scData_GEP.csv"), index_col=0)
    adata_sc = ad.AnnData(df.T)
    # Load cell types as annotation, if available
    cells = pd.read_csv(os.path.join(dataset_folder, "scData_Cells.csv"))
    for cell_type_key in cell_type_keys:
        if cell_type_key in cells.columns:
            adata_sc.obs[cell_type_key] = cells[cell_type_key].values
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



if __name__ == "__main__":

    dataset_from = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP"
    dataset_to = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP_HVG_2000"
    N = 2000

    """ Create dataset from full scRNA + ST dataset """

    # 1. Load scRNA & ST datasets
    scData = load_sc_adata(dataset_from, ["cellType", "cellTypeMinor"])  # C x G
    print("scRNA Data loaded")
    stData = load_st_adata(dataset_from)  # S x G
    print("ST Data loaded")

    # 2. Split genes into shared and non-shared genes
    sharedGenes = scData.var_names.intersection(stData.var_names)

    # 3. Create list of scRNA genes to keep (top N Highly Variable Genes)
    # Restrict HVG selection to genes shared between scRNA and ST
    scDataLog = scData.copy()
    sc.pp.normalize_total(scDataLog)
    sc.pp.log1p(scDataLog)

    # Use scanpy to compute highly variable genes, requesting top N
    # If there are fewer shared genes than N, scanpy will select up to that number
    sc.pp.highly_variable_genes(scDataLog, n_top_genes=N, inplace=True)

    # Collect HVG gene names (var_names where 'highly_variable' is True)
    assert 'highly_variable' in scDataLog.var.columns
    hvgs = list(scDataLog.var_names[scDataLog.var['highly_variable'].values])

    if len(hvgs) == 0:
        raise RuntimeError("No HVGs selected — check input data and N value")

    for sharedGene in sharedGenes:
        if sharedGene not in hvgs:
            print(f"Warning: Shared gene {sharedGene} not selected as HVG - added manually")
            hvgs.append(sharedGene)
    print(f"Appended {len(hvgs) - N} shared genes to HVG list. Total genes selected:", len(hvgs))

    # 4. Create mini scRNA dataset
    scData_HVG = scData[:, hvgs]

    """ Save smaller dataset to disk """

    # 1. Save scData_Cells.csv
    scCells_df = pd.DataFrame(index=scData_HVG.obs_names)
    for col in scData_HVG.obs.columns:
        scCells_df[col] = scData_HVG.obs[col].values
    scCells_df.index.name = "cellID"
    scCells_df.to_csv(os.path.join(dataset_to, "scData_Cells.csv"))

    # 2. Save scData_Genes.csv
    scGenes_df = pd.DataFrame(index=scData_HVG.var_names)
    scGenes_df.index.name = "geneID"
    scGenes_df.to_csv(os.path.join(dataset_to, "scData_Genes.csv"))

    # 3. Save scData_GEP.csv
    scData_HVG = scData_HVG.copy().transpose()
    scData_HVG_df = pd.DataFrame(
        data=scData_HVG.X.toarray() if issparse(scData_HVG.X) else scData_HVG.X,
        index=scData_HVG.obs_names,
        columns=scData_HVG.var_names,
    )
    scData_HVG_df.index.name = "GEP"
    scData_HVG_df.to_csv(os.path.join(dataset_to, "scData_GEP.csv"))

    # 4. Simply copy ST files (no change in genes)
    stData_files = ["stData_Spots.csv", "stData_GEP.csv", "stData_Genes.csv"]
    for file in stData_files:
        src = os.path.join(dataset_from, file)
        dst = os.path.join(dataset_to, file)
        if os.path.exists(src):
            os.system(f"cp {src} {dst}")
