import pandas as pd

major_types = [
    'Hippocampus', 'Internal Capsule Caudoputamen',
    'Layer 2-3 lateral', 'Layer 2-3 medial', 'Layer 3-4', 'Layer 4',
    'Layer 5', 'Layer 6', 'Pia Layer 1', 'Ventricle', 'White matter'
]

df = pd.read_csv("/Users/domi/Downloads/GSE185862_metadata_ssv4_SSp.csv")

# Prüfe, in welcher Spalte die Typen vorkommen
for col in df.columns:
    vals = set(df[col].astype(str).unique())
    if any(t in vals for t in major_types):
        print(f"Spalte '{col}' enthält Major Cell Types.")

# Beispiel: Filter auf die Spalte 'cortical_layer_label'
filtered = df[df["cortical_layer_label"].isin(major_types)]
print(f"Gefilterte Zellanzahl: {len(filtered)}")

# Optional: Schreibe die gefilterten Zellen als neue CSV
filtered.to_csv("/Users/domi/Downloads/GSE185862_metadata_ssv4_SSp_major_types.csv", index=False)