#!/usr/bin/env python3
"""
createGepMatrixFromSparse.py

Read a dataset folder containing:
 - stData_Spots.csv
 - stData_Genes.csv
 - stData_Sparse.mtx

and produce a CSV where:
 - position 0,0 contains the literal 'GEP'
 - the first row (columns) are the gene names
 - the first column (rows) are the spot names

Usage:
    python createStGepMatrixFromSparse.py /path/to/01_HumanBreastCancer_CID4290 \
        --out /path/to/output.csv

Dependencies:
    pandas, scipy

The script will try to infer whether the MTX is genes x cells or cells x genes
by comparing shapes to the lengths of the gene and cell lists and transpose if needed.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

try:
    from scipy.io import mmread
    from scipy.sparse import issparse
except Exception:
    mmread = None  # type: ignore


def read_names(path: str) -> List[str]:
    """Read a simple CSV of names and return a list of strings.

    This reads the first column of the CSV regardless of headers.
    """
    # Read with header (first row is a header). Many scData_*.csv files have
    # a header row followed by one column with the names.
    df = pd.read_csv(path)
    if df.shape[1] >= 1:
        names = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    else:
        # Fallback: use the index if there are no columns (unlikely)
        names = df.index.astype(str).tolist()
    return names


def load_mtx(path: str):
    if mmread is None:
        raise RuntimeError("scipy is required to read .mtx files (install scipy)")
    mat = mmread(path)
    return mat


def main(argv=None) -> int:
    # parser = argparse.ArgumentParser(
    #     description="Create GEP CSV from scData_* files in a dataset folder"
    # )
    # parser.add_argument("dataset_dir", help="Path to dataset folder (contains scData_*.csv and .mtx)")
    # parser.add_argument("--out", "-o", help="Output CSV path. Defaults to <dataset_dir>_GEP.csv")
    # args = parser.parse_args(argv)

    # dataset_dir = os.path.abspath(args.dataset_dir)
    # out_path = args.out if args.out else os.path.join(os.getcwd(), os.path.basename(dataset_dir) + "_GEP.csv")

    dataset_dir = "/MPA_DATA/01_HumanBreastCancer_CID44971"
    out_path = os.path.join(os.getcwd(), os.path.basename(dataset_dir) + "_GEP.csv")

    if not os.path.isdir(dataset_dir):
        print(f"Error: dataset_dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    spots_path = os.path.join(dataset_dir, "stData_Spots.csv")
    genes_path = os.path.join(dataset_dir, "stData_Genes.csv")
    mtx_path = os.path.join(dataset_dir, "stData_Sparse.mtx")

    for p in (spots_path, genes_path, mtx_path):
        if not os.path.exists(p):
            print(f"Error: required file missing: {p}", file=sys.stderr)
            return 3

    spots = read_names(spots_path)
    genes = read_names(genes_path)

    print(f"Read {len(spots)} spots and {len(genes)} genes")

    mat = load_mtx(mtx_path)

    # If sparse, keep as sparse until conversion
    if hasattr(mat, "shape"):
        r, c = mat.shape
        print(f"MTX shape: {mat.shape}")
    else:
        mat = np.asarray(mat)
        r, c = mat.shape

    # We expect final CSV to have rows=spots, columns=genes (data shape: n_spots x n_genes)
    n_spots = len(spots)
    n_genes = len(genes)

    # Determine orientation and transpose if necessary
    if (r, c) == (n_genes, n_spots):
        print("MTX appears to be genes x spots -> transposing to spots x genes")
        if issparse(mat):
            mat = mat.T.tocsr()
        else:
            mat = mat.T
    elif (r, c) == (n_spots, n_genes):
        print("MTX appears to be spots x genes -> keeping orientation")
    else:
        # If shape doesn't match, try to coerce or raise an informative error
        msg = (
            f"MTX shape {mat.shape} does not match lengths: "
            f"genes={n_genes}, spots={n_spots}."
        )
        print("Warning: " + msg, file=sys.stderr)

        # Try transpose and re-check
        if (c, r) == (n_genes, n_spots):
            print("After transposing the MTX it matches gene/spot counts -> transposing")
            if issparse(mat):
                mat = mat.T.tocsr()
            else:
                mat = mat.T
        else:
            print("Cannot infer correct orientation. Aborting.", file=sys.stderr)
            return 4

    # Convert to dense (may be memory heavy for very large matrices)
    if issparse(mat):
        arr = mat.toarray()
    else:
        arr = np.asarray(mat)

    # Sanity check shapes
    if arr.shape != (n_spots, n_genes):
        print(f"Error: final data shape {arr.shape} != (spots, genes) ({n_spots},{n_genes})", file=sys.stderr)
        return 5

    # Build DataFrame with index=spots and columns=genes
    df = pd.DataFrame(arr, index=spots, columns=genes)

    # Use index_label='GEP' so that position 0,0 contains the string 'GEP'
    df.to_csv(out_path, index=True, index_label="GEP")

    print(f"Wrote CSV to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


