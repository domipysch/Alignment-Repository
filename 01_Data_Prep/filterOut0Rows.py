import pandas as pd
import argparse
import os

def filter_zero_rows(input_path, output_path=None):
    df = pd.read_csv(input_path, index_col=0, header=0)
    # Filter: Nur Reihen (rows) behalten, die nicht ausschließlich Nullen sind
    filtered = df.loc[~(df == 0).all(axis=1), :]
    # Output-Pfad bestimmen
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.getcwd(), f"{base}.filtered{ext}")
    filtered.to_csv(output_path)
    print(f"Gefilterte CSV geschrieben nach: {output_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Filter rows with only zeros from a CSV file.")
    # parser.add_argument("input", help="Pfad zur Eingabe-CSV")
    # parser.add_argument("--out", "-o", help="Pfad zur Ausgabe-CSV (optional)")
    # args = parser.parse_args()

    filter_zero_rows(
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_GEP.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/scData_GEP_New.csv",
    )

    filter_zero_rows(
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/stData_GEP.csv",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/02_MouseCortex/stData_GEP_New.csv",
    )
