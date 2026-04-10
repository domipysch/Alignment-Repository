import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse


def is_intable(x) -> bool:
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
    if is_intable(x):
        return int(x)
    raise ValueError("not intable")


def validate_dataset(
    name: str, folder: Path, index_row: pd.Series
) -> tuple[list[str], list[str]]:
    """
    Validate a single dataset folder against the expected h5ad structure.

    Args:
        name: Dataset name (used in error messages).
        folder: Path to the dataset folder.
        index_row: Corresponding row from index.csv with metadata (counts, flags).

    Returns:
        A tuple (errors, warnings), each a list of message strings.
    """
    errors: list[str] = []
    warns: list[str] = []

    if not folder.exists() or not folder.is_dir():
        errors.append(f"Folder missing: {folder}")
        return errors, warns

    sc_path = folder / "sc.h5ad"
    st_path = folder / "st.h5ad"

    if not sc_path.exists():
        errors.append("Missing file: sc.h5ad")
    if not st_path.exists():
        errors.append("Missing file: st.h5ad")

    if errors:
        return errors, warns

    # --- Load sc.h5ad ---
    try:
        adata_sc = ad.read_h5ad(sc_path)
    except Exception as e:
        errors.append(f"sc.h5ad: cannot be read ({e})")
        return errors, warns

    # --- Load st.h5ad ---
    try:
        adata_st = ad.read_h5ad(st_path)
    except Exception as e:
        errors.append(f"st.h5ad: cannot be read ({e})")
        return errors, warns

    # --- Validate sc.h5ad ---

    # Check for NaN or negative values in scRNA expression
    X_sc = adata_sc.X
    if issparse(X_sc):
        X_sc = X_sc.toarray()
    else:
        X_sc = np.asarray(X_sc)

    if np.isnan(X_sc).any():
        errors.append("sc.h5ad: NaN values found in X")
    elif (X_sc < 0).any():
        errors.append("sc.h5ad: negative values found in X")

    # Check cell type annotation if flag is set
    ctype_flag = str(index_row.get("CellTypeAnnotationsExist", "")).strip()
    if ctype_flag not in ("", "0", "False", "false", "None"):
        if "cellType" not in adata_sc.obs.columns:
            warns.append(
                "CellTypeAnnotationsExist is set, but 'cellType' not found in sc.h5ad obs"
            )

    # --- Validate st.h5ad ---

    # Check obsm["spatial"]
    if "spatial" not in adata_st.obsm:
        errors.append("st.h5ad: obsm['spatial'] is missing")
    else:
        spatial = adata_st.obsm["spatial"]
        if spatial.shape != (adata_st.n_obs, 2):
            errors.append(
                f"st.h5ad: obsm['spatial'] shape {spatial.shape} != ({adata_st.n_obs}, 2)"
            )

    # Check for NaN or negative values in ST expression
    X_st = adata_st.X
    if issparse(X_st):
        X_st = X_st.toarray()
    else:
        X_st = np.asarray(X_st)

    if np.isnan(X_st).any():
        errors.append("st.h5ad: NaN values found in X")
    elif (X_st < 0).any():
        errors.append("st.h5ad: negative values found in X")

    # --- Cross-check counts from index.csv ---
    checks = [
        ("scData_CellCount", "cells", lambda: adata_sc.n_obs),
        ("scData_GeneCount", "sc genes", lambda: adata_sc.n_vars),
        ("stData_SpotCount", "spots", lambda: adata_st.n_obs),
        ("stData_GeneCount", "st genes", lambda: adata_st.n_vars),
    ]
    for idx_col, label, getter in checks:
        val = index_row.get(idx_col, "")
        if is_intable(val):
            expected = to_int_safe(val)
            actual = getter()
            if expected != actual:
                warns.append(
                    f"index.csv {idx_col}={expected} != actual {label}={actual}"
                )

    return errors, warns


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Validates datasets listed in index.csv (h5ad format)"
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

    index_path: Path = args.index
    if not index_path.exists():
        print(f"ERROR: index.csv not found: {index_path}", file=sys.stderr)
        sys.exit(2)

    try:
        index_df = pd.read_csv(index_path, dtype=str).fillna("")
    except Exception as e:
        print(f"ERROR: index.csv cannot be read: {e}", file=sys.stderr)
        sys.exit(2)

    if args.data_root:
        root = args.data_root
    else:
        root = index_path.parent

    overall_errors: dict[str, list[str]] = {}
    overall_warns: dict[str, list[str]] = {}

    for _, row in index_df.iterrows():
        name = row.get("Name", "").strip()
        if not name:
            continue

        folder = root / name
        errors, warns = validate_dataset(name, folder, row)
        if errors:
            overall_errors[name] = errors
        if warns:
            overall_warns[name] = warns

        status = (
            "OK"
            if (not errors and not warns)
            else ("WARN" if (not errors and warns) else "ERROR")
        )
        print(f"{name}: {status} (errors={len(errors)}, warns={len(warns)})")

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
    Validates datasets listed in index.csv against the h5ad format:
    - Existence of sc.h5ad and st.h5ad
    - st.h5ad: obsm['spatial'] shape
    - Both: X values >= 0, no NaNs
    - If CellTypeAnnotationsExist is set: verifies 'cellType' column in sc.h5ad obs
    - Count cross-checks against index.csv

    Usage: python validate_database.py --index /path/to/index.csv --data-root /path/to/data_root
    Exit code != 0 on errors.
    """
    main()
