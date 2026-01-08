import pandas as pd
from pathlib import Path
import argparse


def to_uppercase_genes(csv_path: str) -> str:
    """
    Lade eine Genes-CSV (einspaltig oder mit mehreren Spalten),
    konvertiere die Gen-IDs in der Gen-Spalte zu Uppercase und
    schreibe die Datei zurück (überschreibt die Eingabedatei).
    Gibt den Pfad zur geschriebenen Datei zurück.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    # Versuche generisch zu lesen
    df = pd.read_csv(p, dtype=str, keep_default_na=False)

    # Bestimme die Gen-Spalte: bevorzugt 'geneID' (falls vorhanden), sonst erste Spalte
    if "geneID" in df.columns:
        gid_col = "geneID"
    elif "gene" in df.columns:
        gid_col = "gene"
    else:
        gid_col = df.columns[0]

    # Uppercase konvertieren (nur auf echte Strings anwenden)
    df[gid_col] = df[gid_col].astype(str).str.strip().str.upper()

    # Überschreibe die Eingabedatei
    df.to_csv(p, index=False)

    return str(p)


def to_uppercase_genes_gep(gep_csv_path: str) -> str:
    """
    Lade eine GEP-CSV (typischerweise Zeilen = Genes, Spalten = Spots).
    Unterstützt zwei Formate:
      - Gene sind als Index (erste Spalte wurde beim Speichern als Index geschrieben)
      - Gene sind in der ersten Spalte (normaler CSV)
    Konvertiert alle Gen-IDs zu Uppercase und überschreibt die Eingabedatei.
    Gibt den Pfad zur geschriebenen Datei zurück.
    """
    p = Path(gep_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"GEP-Datei nicht gefunden: {gep_csv_path}")

    print("Loading")
    # Versuche zuerst, die Datei mit dem ersten Feld als Index zu lesen
    df = pd.read_csv(p, index_col=0, header=0)
    print("Loaded")

    # Index in Uppercase
    df.index = df.index.astype(str).str.strip().str.upper()
    print("Write")
    df.to_csv("/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_GEP_Out.csv", index=True, header=True)

    return str(p)


def uppercase_file_by_chars(input_path: str, out_path: str | None = None, encoding: str = "utf-8") -> str:
    """
    Liest die Datei zeilenweise und ersetzt alle kleinen Buchstaben durch grosse (char-wise).
    Wenn out_path None ist, wird die Originaldatei atomar ersetzt (via temporäre Datei).
    Gibt den Pfad zur geschriebenen Datei zurück.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {input_path}")

    out_p = Path(out_path)
    temp_p = out_p
    # ensure parent exists
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("r", encoding=encoding, errors="replace") as fin, temp_p.open("w", encoding=encoding) as fout:
        for line in fin:
            # convert lowercase characters to uppercase, leave other chars as-is
            fout.write("".join(ch.upper() if ch.islower() else ch for ch in line))


if __name__ == "__main__":

    uppercase_file_by_chars(
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_GEP.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_GEP_Uppercase.csv",
    )

    quit()

    base_path = "/MPA_DATA"

    folders = [
        # "01_HumanBreastCancer_CID4290",
        # "01_HumanBreastCancer_CID4465",
        # "01_HumanBreastCancer_CID4535",
        # "01_HumanBreastCancer_CID44971",
        # "03_MouseSSP",
        # "04_ColorectalCancer",
        "02_MouseCortex",
    ]

    for folder in folders:
        # csv_path = f"{base_path}/{folder}/scData_Genes.csv"
        # out = to_uppercase_genes(csv_path)
        # print(f"Gen-IDs uppercased und Datei überschrieben: {out}")
        #
        # csv_path = f"{base_path}/{folder}/stData_Genes.csv"
        # out = to_uppercase_genes(csv_path)
        # print(f"Gen-IDs uppercased und Datei überschrieben: {out}")

        # gep_path = f"{base_path}/{folder}/scData_GEP.csv"
        # out = to_uppercase_genes_gep(gep_path)
        # print(f"GEP-Gen-IDs uppercased und Datei überschrieben: {out}")

        gep_path = f"{base_path}/{folder}/stData_GEP.csv"
        out = to_uppercase_genes_gep(gep_path)
        print(f"GEP-Gen-IDs uppercased und Datei überschrieben: {out}")
