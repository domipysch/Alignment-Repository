"""
validate_database.py

Nutzt pandas/pathlib, iteriert alle Einträge in index.csv und validiert pro Dataset:
- Existenz der 6 erwarteten Dateien
- scData: Spalten/IDs/Anzahl/Reihenfolge/GEP >= 0
- stData: Spalten/IDs/Anzahl/Reihenfolge/GEP >= 0
- Falls CellTypeAnnotationsExist == 1: prüfe 'cellType' in scData_GEP columns
- Vergleiche Anzahl-Angaben aus index.csv, falls integer

Aufruf:
  python validate_database.py --index /path/to/index.csv --data-root /path/to/data_root

Kurze, maschinenfreundliche Ausgaben; exit code != 0 bei Fehlern.
"""

import argparse
import sys
import time
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

def read_csv_strict(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Kann CSV nicht lesen: {path} ({e})")

def is_intable(x):
    try:
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, str):
            int(x)
            return True
    except Exception:
        return False
    return False

def to_int_safe(x):
    if is_intable(x):
        return int(x)
    raise ValueError("not intable")

def validate_dataset(name: str, folder: Path, index_row: pd.Series):
    errors = []
    warns = []

    # Prüfe existenz folder
    if not folder.exists() or not folder.is_dir():
        errors.append(f"Ordner fehlt: {folder}")
        return errors, warns

    # Prüfe Dateien vorhanden
    files = {}
    for key, fname in EXPECTED_FILES.items():
        p = folder / fname
        if not p.exists():
            errors.append(f"Fehlende Datei: {fname}")
        else:
            files[key] = p

    if errors:
        return errors, warns

    # Laden scData_Cells
    sc_cells = read_csv_strict(files["sc_cells"])
    if "cellID" not in sc_cells.columns:
        errors.append("scData_Cells.csv: fehlende Spalte 'cellID'")

    # sc_genes: genau eine Spalte 'geneID'
    sc_genes = read_csv_strict(files["sc_genes"])
    if list(sc_genes.columns) != ["geneID"]:
        errors.append(f"scData_Genes.csv: erwartet genau eine Spalte 'geneID', gefunden: {list(sc_genes.columns)}")

    # sc_gep: Index geneIDs (erste Spalte), Spalten cellIDs
    try:
        sc_gep = read_csv_strict(files["sc_gep"], index_col=0)
    except Exception as e:
        errors.append(f"scData_GEP.csv: kann nicht als GEP geladen werden ({e})")
        return errors, warns

    # Optional: wenn CellTypeAnnotationsExist -> 'cellType' in sc_gep.columns?
    ctype_flag = str(index_row.get("CellTypeAnnotationsExist", "")).strip()
    if ctype_flag not in ("", "0", "False", "false", "None"):
        # treat as existing if '1' or non-empty
        if ("cellType" not in sc_cells.columns):
            warns.append("CellTypeAnnotationsExist gesetzt, aber 'cellType' nicht in scData_Cells Spalten gefunden")

    # Prüfe cellIDs Übereinstimmung und Reihenfolge
    if "cellID" in sc_cells.columns:
        cells_list = sc_cells["cellID"].astype(str).tolist()
        gep_cells = list(map(str, sc_gep.columns.tolist()))
        if cells_list != gep_cells:
            # prüfe ob gleiche Menge aber andere Reihenfolge
            if set(cells_list) == set(gep_cells):
                warns.append("Zellen in scData_Cells und scData_GEP sind identisch, Reihenfolge jedoch unterschiedlich")
            else:
                missing_in_gep = [c for c in cells_list if c not in gep_cells]
                missing_in_cells = [c for c in gep_cells if c not in cells_list]
                if missing_in_gep:
                    errors.append(f"Zellen in scData_Cells fehlen in scData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}")
                if missing_in_cells:
                    errors.append(f"Zellen in scData_GEP fehlen in scData_Cells: {missing_in_cells[:5]}{'...' if len(missing_in_cells)>5 else ''}")

    # Prüfe Gene Übereinstimmung und Reihenfolge für sc
    sc_genes_list = sc_genes["geneID"].astype(str).tolist()
    gep_genes = list(map(str, sc_gep.index.astype(str).tolist()))
    if sc_genes_list != gep_genes:
        if set(sc_genes_list) == set(gep_genes):
            warns.append("scData_Genes und scData_GEP Genlisten identisch, Reihenfolge unterschiedlich")
        else:
            missing_in_gep = [g for g in sc_genes_list if g not in gep_genes]
            missing_in_genes = [g for g in gep_genes if g not in sc_genes_list]
            if missing_in_gep:
                errors.append(f"scData_Genes fehlen in scData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}")
            if missing_in_genes:
                errors.append(f"scData_GEP enthält Gene, die nicht in scData_Genes sind: {missing_in_genes[:5]}{'...' if len(missing_in_genes)>5 else ''}")

    # scData GEP Werte >= 0
    try:
        # coerce to numeric matrix
        numeric = sc_gep.apply(pd.to_numeric, errors="coerce")
        if numeric.isnull().values.any():
            errors.append("scData_GEP: nicht-numerische Werte gefunden (NaN nach Konvertierung)")
        else:
            if (numeric.values < 0).any():
                errors.append("scData_GEP: negative Werte gefunden")
    except Exception as e:
        errors.append(f"scData_GEP: Fehler bei numerischer Prüfung ({e})")

    # --- stData ---
    st_genes = read_csv_strict(files["st_genes"])
    if list(st_genes.columns) != ["geneID"]:
        errors.append(f"stData_Genes.csv: erwartet genau eine Spalte 'geneID', gefunden: {list(st_genes.columns)}")

    st_spots = read_csv_strict(files["st_spots"])
    expected_spot_cols = ["spotID", "cArray0", "cArray1"]
    if list(st_spots.columns) != expected_spot_cols:
        errors.append(f"stData_Spots.csv: erwartet Spalten {expected_spot_cols}, gefunden: {list(st_spots.columns)}")

    st_gep = read_csv_strict(files["st_gep"], index_col=0)
    # Prüfe spots
    if "spotID" in st_spots.columns:
        spots_list = st_spots["spotID"].astype(str).tolist()
        gep_spots = list(map(str, st_gep.columns.tolist()))
        if spots_list != gep_spots:
            if set(spots_list) == set(gep_spots):
                warns.append("Spots in stData_Spots und stData_GEP identisch, Reihenfolge jedoch unterschiedlich")
            else:
                missing_in_gep = [s for s in spots_list if s not in gep_spots]
                missing_in_spots = [s for s in gep_spots if s not in spots_list]
                if missing_in_gep:
                    errors.append(f"Spots in stData_Spots fehlen in stData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}")
                if missing_in_spots:
                    errors.append(f"Spots in stData_GEP fehlen in stData_Spots: {missing_in_spots[:5]}{'...' if len(missing_in_spots)>5 else ''}")

    # Prüfe Gene Übereinstimmung und Reihenfolge für st
    st_genes_list = st_genes["geneID"].astype(str).tolist()
    st_gep_genes = list(map(str, st_gep.index.astype(str).tolist()))
    if st_genes_list != st_gep_genes:
        if set(st_genes_list) == set(st_gep_genes):
            warns.append("stData_Genes und stData_GEP Genlisten identisch, Reihenfolge unterschiedlich")
        else:
            missing_in_gep = [g for g in st_genes_list if g not in st_gep_genes]
            missing_in_genes = [g for g in st_gep_genes if g not in st_genes_list]
            if missing_in_gep:
                errors.append(f"stData_Genes fehlen in stData_GEP: {missing_in_gep[:5]}{'...' if len(missing_in_gep)>5 else ''}")
            if missing_in_genes:
                errors.append(f"stData_GEP enthält Gene, die nicht in stData_Genes sind: {missing_in_genes[:5]}{'...' if len(missing_in_genes)>5 else ''}")

    # stData GEP Werte >= 0
    try:
        numeric_st = st_gep.apply(pd.to_numeric, errors="coerce")
        if numeric_st.isnull().values.any():
            errors.append("stData_GEP: nicht-numerische Werte gefunden (NaN nach Konvertierung)")
        else:
            if (numeric_st.values < 0).any():
                errors.append("stData_GEP: negative Werte gefunden")
    except Exception as e:
        errors.append(f"stData_GEP: Fehler bei numerischer Prüfung ({e})")

    # Vergleich mit index.csv Angaben (sofern int)
    checks = [
        ("scData_CellCount", "cells", lambda: len(sc_cells) if "cellID" in sc_cells.columns else None),
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
                warns.append(f"{label}: actual value unknown, cannot compare to index.csv {idx_col}")
            else:
                if expected != actual:
                    warns.append(f"index.csv {idx_col}={expected} != actual {label}={actual}")

    return errors, warns


def main():
    parser = argparse.ArgumentParser(description="Validates datasets listed in index.csv")
    parser.add_argument("--index", "-i", type=Path, default=Path("/Users/domi/Dev/MPA_Workspace/MPA_DATA/index.csv"))
    parser.add_argument("--data-root", "-d", type=Path, default=None,
                        help="Root folder where dataset folders (Name) live. Wenn nicht angegeben, versucht einige Kandidaten.")
    args = parser.parse_args()

    index_path = args.index
    if not index_path.exists():
        print(f"ERROR: index.csv nicht gefunden: {index_path}", file=sys.stderr)
        sys.exit(2)

    try:
        index_df = pd.read_csv(index_path, dtype=str).fillna("")
    except Exception as e:
        print(f"ERROR: index.csv kann nicht gelesen werden: {e}", file=sys.stderr)
        sys.exit(2)

    # Daten-Root Folder für Kandidaten
    if args.data_root:
        root = args.data_root
    else:
        root = index_path.parent

    overall_errors = {}
    overall_warns = {}

    for _, row in index_df.iterrows():

        name = row.get("Name", "").strip()
        if not name:
            continue

        # finde Folder: suche in roots nach ersten match where folder exists
        folder = root / name
        errors, warns = validate_dataset(name, folder if folder else Path(index_path.parent), row)
        if errors:
            overall_errors[name] = errors
        if warns:
            overall_warns[name] = warns

        # kurze Ausgabe pro dataset
        status = "OK" if (not errors and not warns) else ("WARN" if (not errors and warns) else "ERROR")
        print(f"{name}: {status} (errors={len(errors)}, warns={len(warns)})")

    time.sleep(1)

    # Zusammenfassung
    if overall_warns:
        print("\nWARNINGS:")
        for name, w in overall_warns.items():
            print(f"- {name}:")
            for msg in w:
                print(f"    - {msg}")

    if overall_errors:
        print("\nERRORS:", file=sys.stderr)
        for name, e in overall_errors.items():
            print(f"- {name}:", file=sys.stderr)
            for msg in e:
                print(f"    - {msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAlle Datensätze validiert (keine Fehler).")
        sys.exit(0)


if __name__ == "__main__":
    main()
