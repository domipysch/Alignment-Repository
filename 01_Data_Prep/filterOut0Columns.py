import pandas as pd
import argparse
import os

def filter_zero_columns(input_path, output_path=None):
    df = pd.read_csv(input_path, index_col=0)
    # Filter: Nur Spalten behalten, die nicht ausschließlich Nullen sind
    filtered = df.loc[:, ~(df == 0).all()]
    # Output-Pfad bestimmen
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.getcwd(), f"{base}.filtered{ext}")
    filtered.to_csv(output_path)
    print(f"Gefilterte CSV geschrieben nach: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter columns with only zeros from a CSV file.")
    parser.add_argument("input", help="Pfad zur Eingabe-CSV")
    parser.add_argument("--out", "-o", help="Pfad zur Ausgabe-CSV (optional)")
    args = parser.parse_args()
    filter_zero_columns(args.input, args.out)

