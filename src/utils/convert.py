import pandas as pd
import scanpy as sc


def pd_df_to_anndata(df: pd.DataFrame, transpose: bool) -> sc.AnnData:
    if transpose:
        df = df.T
    return sc.AnnData(
        X=df.values,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )


def anndata_to_pd_df(adata: sc.AnnData, transpose: bool) -> pd.DataFrame:
    df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    if transpose:
        df = df.T
    return df
