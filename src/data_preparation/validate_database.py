import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED_FILES = {
    "sc_cells": "scData_Cells.csv",
    "sc_genes": "scData_Genes.csv",
    "sc_gep": "scData_GEP.csv",
    "st_genes": "stData_Genes.csv",
    "st_spots": "stData_Spots.csv",
    "st_gep": "stData_GEP.csv",
}


def read_csv_strict(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file, raising a RuntimeError on failure.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional keyword arguments forwarded to pd.read_csv.

    Returns:
        The parsed DataFrame.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV: {path} ({e})")


def is_intable(x) -> bool:
    """
    Return True if x can be interpreted as an integer.

    Args:
        x: Value to check (int, np.integer, or str).
    """
    try:
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, str):
            int(x)
            return True
    except Exception:
        return False
    return False


def to_int_safe(x) -> int:
    """
    Convert x to int, raising ValueError if not possible.

    Args:
        x: Value to convert.
    """
    if is_intable(x):
        return int(x)
    raise ValueError("not intable")


def validate_dataset(
    name: str, folder: Path, index_row: pd.Series
) -> tuple[list[str], list[str]]:
    """
    Validate a single dataset folder against the expected file structure and contents.

    Args:
        name: Dataset name (used in error messages).
        folder: Path to the dataset folder.
        index_row: Corresponding row from index.csv with metadata (counts, flags).

    Returns:
        A tuple (errors, warnings), each a list of message strings.
    """
    errors: list[str] = []
    warns: list[str] = []

    # Check if folder exists
    if not folder.exists() or not folder.is_dir():
        errors.append(f"Folder missing: {folder}")
        return errors, warns

    # Check if all expected files are present
    files = {}
    for key, fname in EXPECTED_FILES.items():
        p = folder / fname
        if not p.exists():
            errors.append(f"Missing file: {fname}")
        else:
            files[key] = p

    if errors:
        return errors, warns

    # Load scData_Cells
    sc_cells = read_csv_strict(files["sc_cells"])
    if "cellID" not in sc_cells.columns:
        errors.append("scData_Cells.csv: missing column 'cellID'")

    # sc_genes: must have exactly one column 'geneID'
    sc_genes = read_csv_strict(files["sc_genes"])
    if list(sc_genes.columns) != ["geneID"]:
        errors.append(
            f"scData_Genes.csv: expected exactly one column 'geneID', found: {list(sc_genes.columns)}"
        )

    # sc_gep: index = geneIDs (first column), columns = cellIDs
    try:
        sc_gep = read_csv_strict(files["sc_gep"], index_col=0)
    except Exception as e:
        errors.append(f"scData_GEP.csv: cannot be loaded as GEP ({e})")
        return errors, warns

    # If CellTypeAnnotationsExist is set, verify 'cellType' column is present
    ctype_flag = str(index_row.get("CellTypeAnnotationsExist", "")).strip()
    if ctype_flag not in ("", "0", "False", "false", "None"):
        # treat as existing if '1' or non-empty
        if "cellType" not in sc_cells.columns:
            warns.append(
                "CellTypeAnnotationsExist is set, but 'cellType' not found in scData_Cells columns"
            )

    # Check cellID consistency and order between scData_Cells and scData_GEP
    if "cellID" in sc_cells.columns:
        cells_list = sc_cells["cellID"].astype(str).tolist()
        gep_cells = list(map(str, sc_gep.columns.tolist()))
        if cells_list != gep_cells:
            # same set but different order
            if set(cells_list) == set(gep_cells):
                warns.append(
                    "Cell IDs in scData_Cells and scData_GEP are identical but in different order"
                )
            else:
                missing_in_gep = [c for c in cells_list if c not in gep_cells]
                missing_in_cells = [c for c in gep_cells if c not in cells_list]
                if missing_in_gep:
                    errors.append(
                        f"Cells in scData_Cells missing from scData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}"
                    )
                if missing_in_cells:
                    errors.append(
                        f"Cells in scData_GEP missing from scData_Cells: {missing_in_cells[:5]}{'...' if len(missing_in_cells)>5 else ''}"
                    )

    # Check gene consistency and order for sc
    sc_genes_list = sc_genes["geneID"].astype(str).tolist()
    gep_genes = list(map(str, sc_gep.index.astype(str).tolist()))
    if sc_genes_list != gep_genes:
        if set(sc_genes_list) == set(gep_genes):
            warns.append(
                "Gene lists in scData_Genes and scData_GEP are identical but in different order"
            )
        else:
            missing_in_gep = [g for g in sc_genes_list if g not in gep_genes]
            missing_in_genes = [g for g in gep_genes if g not in sc_genes_list]
            if missing_in_gep:
                errors.append(
                    f"Genes in scData_Genes missing from scData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}"
                )
            if missing_in_genes:
                errors.append(
                    f"scData_GEP contains genes not in scData_Genes: {missing_in_genes[:5]}{'...' if len(missing_in_genes)>5 else ''}"
                )

    # scData GEP values must be >= 0
    try:
        # coerce to numeric matrix
        numeric = sc_gep.apply(pd.to_numeric, errors="coerce")
        if numeric.isnull().values.any():
            errors.append("scData_GEP: non-numeric values found (NaN after conversion)")
        else:
            if (numeric.values < 0).any():
                errors.append("scData_GEP: negative values found")
    except Exception as e:
        errors.append(f"scData_GEP: error during numeric check ({e})")

    # --- stData ---
    st_genes = read_csv_strict(files["st_genes"])
    if list(st_genes.columns) != ["geneID"]:
        errors.append(
            f"stData_Genes.csv: expected exactly one column 'geneID', found: {list(st_genes.columns)}"
        )

    st_spots = read_csv_strict(files["st_spots"])
    expected_spot_cols = ["spotID", "cArray0", "cArray1"]
    if list(st_spots.columns) != expected_spot_cols:
        errors.append(
            f"stData_Spots.csv: expected columns {expected_spot_cols}, found: {list(st_spots.columns)}"
        )

    st_gep = read_csv_strict(files["st_gep"], index_col=0)
    # Check spot ID consistency and order
    if "spotID" in st_spots.columns:
        spots_list = st_spots["spotID"].astype(str).tolist()
        gep_spots = list(map(str, st_gep.columns.tolist()))
        if spots_list != gep_spots:
            if set(spots_list) == set(gep_spots):
                warns.append(
                    "Spot IDs in stData_Spots and stData_GEP are identical but in different order"
                )
            else:
                missing_in_gep = [s for s in spots_list if s not in gep_spots]
                missing_in_spots = [s for s in gep_spots if s not in spots_list]
                if missing_in_gep:
                    errors.append(
                        f"Spots in stData_Spots missing from stData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}"
                    )
                if missing_in_spots:
                    errors.append(
                        f"Spots in stData_GEP missing from stData_Spots: {missing_in_spots[:5]}{'...' if len(missing_in_spots)>5 else ''}"
                    )

    # Check gene consistency and order for st
    st_genes_list = st_genes["geneID"].astype(str).tolist()
    st_gep_genes = list(map(str, st_gep.index.astype(str).tolist()))
    if st_genes_list != st_gep_genes:
        if set(st_genes_list) == set(st_gep_genes):
            warns.append(
                "Gene lists in stData_Genes and stData_GEP are identical but in different order"
            )
        else:
            missing_in_gep = [g for g in st_genes_list if g not in st_gep_genes]
            missing_in_genes = [g for g in st_gep_genes if g not in st_genes_list]
            if missing_in_gep:
                errors.append(
                    f"Genes in stData_Genes missing from stData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}"
                )
            if missing_in_genes:
                errors.append(
                    f"stData_GEP contains genes not in stData_Genes: {missing_in_genes[:5]}{'...' if len(missing_in_genes)>5 else ''}"
                )

    # stData GEP values must be >= 0
    try:
        numeric_st = st_gep.apply(pd.to_numeric, errors="coerce")
        if numeric_st.isnull().values.any():
            errors.append("stData_GEP: non-numeric values found (NaN after conversion)")
        else:
            if (numeric_st.values < 0).any():
                errors.append("stData_GEP: negative values found")
    except Exception as e:
        errors.append(f"stData_GEP: error during numeric check ({e})")

    # Cross-check count values from index.csv
    checks = [
        (
            "scData_CellCount",
            "cells",
            lambda: len(sc_cells) if "cellID" in sc_cells.columns else None,
        ),
        ("scData_GeneCount", "genes", lambda: len(sc_genes)),
        ("stData_SpotCount", "spots", lambda: len(st_spots)),
        ("stData_GeneCount", "st_genes", lambda: len(st_genes)),
    ]
    for idx_col, label, getter in checks:
        val = index_row.get(idx_col, "")
        if is_intable(val):
            expected = to_int_safe(val)
            actual = getter()
            if actual is None:
                warns.append(
                    f"{label}: actual value unknown, cannot compare to index.csv {idx_col}"
                )
            else:
                if expected != actual:
                    warns.append(
                        f"index.csv {idx_col}={expected} != actual {label}={actual}"
                    )

    return errors, warns


def main() -> None:

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Validates datasets listed in index.csv"
    )
    parser.add_argument(
        "--index",
        "-i",
        type=Path,
        default=Path("/Users/domi/Dev/MPA_Workspace/MPA_DATA/index.csv"),
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=Path,
        default=None,
        help="Root folder where dataset folders live. Falls back to the index.csv parent directory if not provided.",
    )
    args = parser.parse_args()

    # Get path to index file
    index_path: Path = args.index
    if not index_path.exists():
        print(f"ERROR: index.csv not found: {index_path}", file=sys.stderr)
        sys.exit(2)

    # Load index file
    try:
        index_df = pd.read_csv(index_path, dtype=str).fillna("")
    except Exception as e:
        print(f"ERROR: index.csv cannot be read: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve data root directory
    if args.data_root:
        root = args.data_root
    else:
        root = index_path.parent

    overall_errors: dict[str, list[str]] = {}
    overall_warns: dict[str, list[str]] = {}

    # Iterate over datasets in index.csv and validate each one
    for _, row in index_df.iterrows():

        name = row.get("Name", "").strip()
        if not name:
            continue

        # Find the dataset folder under root
        folder = root / name
        errors, warns = validate_dataset(
            name, folder if folder else Path(index_path.parent), row
        )
        if errors:
            overall_errors[name] = errors
        if warns:
            overall_warns[name] = warns

        # Brief per-dataset status line
        status = (
            "OK"
            if (not errors and not warns)
            else ("WARN" if (not errors and warns) else "ERROR")
        )
        print(f"{name}: {status} (errors={len(errors)}, warns={len(warns)})")

    # Summary
    if overall_warns:
        print("\nWARNINGS:")
        for name, w in overall_warns.items():
            print(f"- {name}:")
            for msg in w:
                print(f"    - {msg}")

    if overall_errors:
        print("\nERRORS:", file=sys.stderr)
        for name, err in overall_errors.items():
            print(f"- {name}:", file=sys.stderr)
            for msg in err:
                print(f"    - {msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll datasets validated successfully (no errors).")
        sys.exit(0)


if __name__ == "__main__":
    """
    Iterates all entries in index.csv and validates each dataset:
    - Existence of the 6 expected files
    - scData/stData: column names, ID consistency, count cross-checks, GEP values >= 0
    - If CellTypeAnnotationsExist is set: verifies 'cellType' column in scData_Cells

    Usage: python validate_database.py --index /path/to/index.csv --data-root /path/to/data_root
    Exit code != 0 on errors.
    """
    main()
