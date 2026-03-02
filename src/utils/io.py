from pathlib import Path
from typing import Optional, Callable
from scipy.sparse import issparse
import pandas as pd
import numpy as np
from anndata import AnnData


def csv_to_anndata(csv_path: Path, transpose: bool) -> AnnData:
    """
    Load a CSV file and return it as an AnnData object.

    Args:
        csv_path: Path to the input CSV file.
        transpose: Whether to transpose the data after loading
    """
    df = pd.read_csv(csv_path, header=0, index_col=0)
    if transpose:
        df = df.T
    X = np.asarray(df.values)
    ad = AnnData(
        X=X, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns)
    )
    ad.obs_names = list(df.index)
    ad.var_names = list(df.columns)
    return ad


def anndata_to_csv(
    adata: AnnData,
    output_path: Path,
    top_left_label: str,
    format_func: Optional[Callable[[float], str]] = None,
    uppercase_var_names: bool = False,
):
    """
    Write an AnnData object to a CSV file.

    Args:
        top_left_label: Value used as the index label (top-left cell of the CSV).
        format_func: Optional element-wise formatting function applied via DataFrame.map.
        uppercase_var_names: If True, obs_names (row index) are uppercased before writing.
    """

    # Extract dense matrix from AnnData (X: obs × var)
    X = adata.X
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    index = list(adata.obs_names)
    if uppercase_var_names:
        index = [s.upper() for s in index]
    columns = list(adata.var_names)

    df = pd.DataFrame(X, index=index, columns=columns)

    # Optional element-wise formatting
    if format_func is not None:
        df_out = df.map(format_func)
    else:
        df_out = df

    # Write CSV with custom top-left label
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=True, index_label=top_left_label)
