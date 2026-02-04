import os
import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import issparse


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
    dataset_to = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_Datasets/03_MouseSSP_Mini"
    target_nr_sc_cells = 15
    target_nr_sc_genes = 30
    target_nr_st_spots = 10
    target_nr_st_genes = 20
    target_nr_shared_genes = 5

    rng = np.random.default_rng()  # Zufallszahlengenerator für reproduzierbares Sampling (ohne festen Seed)

    """ Create mini dataset from full scRNA + ST dataset """

    # 1. Load scRNA & ST datasets
    scData = load_sc_adata(dataset_from, ["cellType", "cellTypeMinor"])  # C x G
    stData = load_st_adata(dataset_from)  # S x G

    # 2. Split genes into shared and non-shared genes
    sharedGenes = scData.var_names.intersection(stData.var_names)
    scNonSharedGenes = scData.var_names.difference(sharedGenes)
    stNonSharedGenes = stData.var_names.difference(sharedGenes)

    # 3. Create list of genes to keep (of length target_nr, half shared, half non-shared)
    scGenesToKeep = list(sharedGenes[:target_nr_shared_genes]) + list(scNonSharedGenes[:target_nr_sc_genes - target_nr_shared_genes])
    if len(stNonSharedGenes) >= target_nr_shared_genes:
        stGenesToKeep = list(sharedGenes[:target_nr_shared_genes]) + list(stNonSharedGenes[:target_nr_st_genes - target_nr_shared_genes])
    else:
        stGenesToKeep = list(sharedGenes[:target_nr_st_genes])

    # 4. Select cells from scRNA data that contain expression for at least one shared gene each (at random, target_nr many)
    # robust handling für dichte und sparse .X
    sc_mat = scData[:, sharedGenes[:target_nr_shared_genes]].X
    if issparse(sc_mat):
        sc_mask = np.asarray(sc_mat.getnnz(axis=1) > 0).reshape(-1)
    else:
        sc_mask = np.any(sc_mat != 0, axis=1).reshape(-1)
    available_sc = scData.obs_names[sc_mask].tolist()
    n_sc = min(len(available_sc), target_nr_sc_cells)
    scCellsToKeep = rng.choice(available_sc, size=n_sc, replace=False).tolist()

    # 5. Select spots from ST data that contain expression for at least one shared gene each (at random, target_nr many)
    st_mat = stData[:, sharedGenes].X
    if issparse(st_mat):
        st_mask = np.asarray(st_mat.getnnz(axis=1) > 0).reshape(-1)
    else:
        st_mask = np.any(st_mat != 0, axis=1).reshape(-1)
    available_st = stData.obs_names[st_mask].tolist()
    n_st = min(len(available_st), target_nr_st_spots)
    stSpotsToKeep = rng.choice(available_st, size=n_st, replace=False).tolist()

    # 6. Create mini scRNA + ST datasets
    scData_mini = scData[scCellsToKeep, scGenesToKeep]
    stData_mini = stData[stSpotsToKeep, stGenesToKeep]

    """ Save mini dataset to disk """

    # 1. Save scData_Cells.csv
    scCells_df = pd.DataFrame(index=scData_mini.obs_names)
    for col in scData_mini.obs.columns:
        scCells_df[col] = scData_mini.obs[col].values
    scCells_df.index.name = "cellID"
    scCells_df.to_csv(os.path.join(dataset_to, "scData_Cells.csv"))

    # 2. Save scData_Genes.csv
    scGenes_df = pd.DataFrame(index=scData_mini.var_names)
    scGenes_df.index.name = "geneID"
    scGenes_df.to_csv(os.path.join(dataset_to, "scData_Genes.csv"))

    # 3. Save stData_Spots.csv
    stSpots_df = pd.DataFrame(index=stData_mini.obs_names)
    spatial = np.asarray(stData_mini.obsm["spatial"])
    # falls (2, n_spots) statt (n_spots, 2), transponieren
    if spatial.shape == (2, stData_mini.n_obs):
        spatial = spatial.T
    stSpots_df[["cArray0", "cArray1"]] = spatial
    stSpots_df.index.name = "spotID"
    stSpots_df.to_csv(os.path.join(dataset_to, "stData_Spots.csv"))

    # 4. Save stData_Genes.csv
    stGenes_df = pd.DataFrame(index=stData_mini.var_names)
    stGenes_df.index.name = "geneID"
    stGenes_df.to_csv(os.path.join(dataset_to, "stData_Genes.csv"))

    # 5. Save scData_GEP.csv
    scData_mini = scData_mini.copy().transpose()
    stData_mini = stData_mini.copy().transpose()

    scData_mini_df = pd.DataFrame(
        data=scData_mini.X.toarray() if issparse(scData_mini.X) else scData_mini.X,
        index=scData_mini.obs_names,
        columns=scData_mini.var_names,
    )
    scData_mini_df.index.name = "GEP"
    scData_mini_df.to_csv(os.path.join(dataset_to, "scData_GEP.csv"))

    # 6. Save stData_GEP.csv
    stData_mini_df = pd.DataFrame(
        data=stData_mini.X.toarray() if issparse(stData_mini.X) else stData_mini.X,
        index=stData_mini.obs_names,
        columns=stData_mini.var_names,
    )
    stData_mini_df.index.name = "GEP"
    stData_mini_df.to_csv(os.path.join(dataset_to, "stData_GEP.csv"))
