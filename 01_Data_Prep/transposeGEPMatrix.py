#!/usr/bin/env python3
"""
transposeGEPMatrix.py

Read a CSV file, transpose it, and write the transposed table to a new CSV.

Behavior:
 - Treat the first row as header (column names).
 - Treat the first column as the index (if present) so index labels are preserved after transpose.
 - If the input has no explicit index column, the script will transpose the raw table.

Usage:
    python transposeGEPMatrix.py /path/to/input.csv
    python transposeGEPMatrix.py /path/to/input.csv --out /path/to/output.csv

Dependencies:
    pandas
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional
import pandas as pd


def transpose_csv(input_path: str, output_path: Optional[str] = None) -> int:
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 2

    # Default output file if not provided
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(input_path))
        parent = os.path.dirname(input_path)
        output_path = os.path.join(parent, f"{base}_Transposed{ext}")

    print("CP0")

    # Try reading with first column as index (common for matrix-like CSVs where first column are row labels)
    try:
        df = pd.read_csv(input_path, index_col=0)
    except Exception as e:
        print(f"Warning: reading with index_col=0 failed: {e} -- trying without index", file=sys.stderr)
        try:
            df = pd.read_csv(input_path)
        except Exception as e2:
            print(f"Error: failed to read CSV: {e2}", file=sys.stderr)
            return 3

    # If the DataFrame is empty, bail out
    if df.shape[0] == 0 or df.shape[1] == 0:
        print(f"Error: input CSV appears empty or malformed: {input_path}", file=sys.stderr)
        return 4

    print("CP1")

    # Transpose
    t = df.transpose()

    # Determine index_label for output CSV: prefer original index name if present, otherwise None
    index_label = None
    # If original df had an index name, that becomes the column header for the index when writing
    if getattr(df.index, "name", None):
        index_label = df.index.name

    # Save CSV; ensure index is written so transposed row labels are present
    try:
        t.to_csv(output_path, index=True, index_label=index_label)
    except Exception as e:
        print(f"Error: failed to write output CSV: {e}", file=sys.stderr)
        return 5

    print(f"Wrote transposed CSV to: {output_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Transpose a CSV file and save the result")
    # parser.add_argument("input", help="Path to input CSV file")
    # parser.add_argument("--out", "-o", help="Output CSV path (default: <input>.transposed.csv)")
    # args = parser.parse_args(argv)

    # return transpose_csv(args.input, args.out)
    transpose_csv(
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_HumanBreastCancer_CID4465/results/dot_spot_by_gene_GEP.csv"
    )


if __name__ == "__main__":
    raise SystemExit(main())


