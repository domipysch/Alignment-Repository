import csv
import os
import sys

def tsv_to_csv(input_path: str, output_path: str) -> None:
    with open(input_path, newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin, delimiter="\t")
        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)  # Standard-Delimiter ist Komma
            for row in reader:
                writer.writerow(row)

if __name__ == "__main__":

    input_path = "/MPA_DATA/00_Preparing_Datasets/04_ColorectalCancer/CytoSPACE_example_colon_cancer_merscope/HumanColonCancerPatient2_ST_expressions_cytospace.tsv"

    base, _ = os.path.splitext(os.path.basename(input_path))
    out_dir = os.path.dirname(input_path) or "."
    output_path = os.path.join(out_dir, f"{base}.csv")

    try:
        tsv_to_csv(input_path, output_path)
    except Exception as e:
        print(f"Fehler beim Konvertieren: {e}", file=sys.stderr)

    print(f"Erfolgreich konvertiert: {output_path}")

