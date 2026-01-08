import sys
import csv
from pathlib import Path

def count_cell_types(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Datei nicht gefunden: {csv_path}", file=sys.stderr)
        return 2
    celltype_set = set()
    celltype_minor_set = set()
    with csv_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        # Prüfe Spalten vorhanden
        if 'cellType' not in reader.fieldnames or 'cellTypeMinor' not in reader.fieldnames:
            print("Erwarte Spalten 'cellType' und 'cellTypeMinor' im CSV.", file=sys.stderr)
            return 3
        for row in reader:
            celltype_set.add((row.get('cellType') or '').strip())
            celltype_minor_set.add((row.get('cellTypeMinor') or '').strip())
    # Entferne ggf. leere Strings aus den Sets
    celltype_set.discard('')
    celltype_minor_set.discard('')
    print(f"Unterschiedliche cellType: {len(celltype_set)}")
    print(f"Unterschiedliche cellTypeMinor: {len(celltype_minor_set)}")
    # kurze Listen ausgeben (nützlich zur Kontrolle)
    print("\ncellType (Beispiele):")
    for v in sorted(celltype_set)[:3]:
        print(f"  {v}")
    print("\ncellTypeMinor (Beispiele):")
    for v in sorted(celltype_minor_set)[:5]:
        print(f"  {v}")
    return 0

def main(path: str):
    count_cell_types(path)

if __name__ == '__main__':
    main(
        "/MPA_DATA/04_ColorectalCancer/scData_Cells.csv"
    )
