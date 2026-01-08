import csv
import os
import sys

def extract_first_column(input_path: str, output_name: str = "stData_Genes.csv") -> None:
    if not os.path.isfile(input_path):
        print(f"Fehler: Eingabedatei nicht gefunden: {input_path}", file=sys.stderr)
        return

    out_dir = os.path.dirname(input_path)
    output_path = os.path.join(out_dir, output_name)

    try:
        with open(input_path, newline="", encoding="utf-8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            first_col = [row[0] for row in reader if row]
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}", file=sys.stderr)
        return

    if not first_col:
        print("Keine Daten in der ersten Spalte gefunden.", file=sys.stderr)
        return

    try:
        with open(output_path, mode="w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)  # Standard-Delimiter ist Komma
            for val in first_col:
                writer.writerow([val])
        print(f"Erfolgreich gespeichert: {output_path}")
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei: {e}", file=sys.stderr)

if __name__ == "__main__":
    extract_first_column(
        "/MPA_DATA/00_Preparing_Datasets/04_ColorectalCancer/CytoSPACE_example_colon_cancer_merscope/HumanColonCancerPatient2_ST_expressions_cytospace.tsv")
