import argparse
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple

def filter_genes_by_gep(
    gep_csv_path: str,
    genes_csv_path: str,
    out_genes_csv_path: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Lade GEP-CSV und Genes-CSV, behalte in Genes-CSV nur Gen-IDs, die in der GEP-Datei vorkommen.
    Rückgabe: (pfad_zur_gespeicherten_datei, anzahl_orig, anzahl_gefiltert)
    - gep_csv_path: Pfad zur GEP-CSV (erste Spalte = Gene oder Index)
    - genes_csv_path: Pfad zur Genes.csv (eine Spalte mit Gen-IDs)
    - out_genes_csv_path: optionaler Ausgabepfad; falls None, wird "<stem>.updated<ext>" im selben Ordner erzeugt
    """

    gep_path = Path(gep_csv_path)
    genes_path = Path(genes_csv_path)

    if not gep_path.exists():
        raise FileNotFoundError(f"GEP file not found: {gep_path}")
    if not genes_path.exists():
        raise FileNotFoundError(f"Genes file not found: {genes_path}")

    # GEP einlesen: versuche Index als Gen-IDs (erste Spalte)
    df_gep = pd.read_csv(gep_path, index_col=0, header=0)
    gep_genes = df_gep.index.astype(str).str.strip().tolist()
    gep_genes_set = set(gep_genes)

    # Genes.csv einlesen
    df_genes = pd.read_csv(genes_path, header=0)

    # Trim whitespace
    df_genes["geneID"] = df_genes["geneID"].astype(str).str.strip()

    n_before = len(df_genes)
    # Filteriere (erhält Reihenfolge aus genes_csv)
    df_filtered = df_genes[df_genes["geneID"].isin(gep_genes_set)]
    n_after = len(df_filtered)

    # Bestimme Ausgabepfad
    if out_genes_csv_path:
        out_path = Path(out_genes_csv_path)
    else:
        out_path = genes_path.with_name(genes_path.stem + ".updated" + genes_path.suffix)

    # Schreibe gefilterte Genes.csv
    df_filtered.to_csv(out_path, index=False)

    return str(out_path), n_before, n_after

# CLI
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Filter Genes.csv by genes present in a GEP CSV.")
    # parser.add_argument("gep", help="Pfad zur GEP-CSV (erste Spalte oder Index = Gene IDs)")
    # parser.add_argument("genes", help="Pfad zur Genes.csv (eine Spalte mit Gen-IDs)")
    # parser.add_argument("--out", "-o", help="optional: Ausgabepfad für gefilterte Genes.csv")
    # parser.add_argument("--col", help="optional: Spaltenname in Genes.csv mit Gen-IDs (Default: geneID oder erste Spalte)")
    # args = parser.parse_args()

    out, before, after = filter_genes_by_gep(
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_GEP_New.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_Genes.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_Genes_New.csv",
    )
    print(f"Gefilterte Genes.csv geschrieben nach: {out}")
    print(f"Anzahl Einträge: vorher={before}, nachher={after}")

    out, before, after = filter_genes_by_gep(
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/stData_GEP_New.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/stData_Genes.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/stData_Genes_New.csv",
    )
    print(f"Gefilterte Genes.csv geschrieben nach: {out}")
    print(f"Anzahl Einträge: vorher={before}, nachher={after}")
