
#!/usr/bin/env python3
"""Count shared gene IDs between two CSV files.

Each file is expected to have a header (e.g. "geneID") and one gene id per line.
The script prints a single integer: the number of gene IDs present in both files.

Usage:
    python3 scripts/count_shared_genes.py file1.csv file2.csv [--ignore-case]

"""
import argparse
import csv
import sys
from pathlib import Path


def read_genes(path: Path):
    s = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            if not gene:
                continue
            # skip typical header label
            if gene.lower() == "geneid":
                continue
            s.add(gene)
    return s


def read_cells(path: Path):
    s = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                print("here")
                continue
            cell = row[0].strip()
            if not cell:
                continue
            # skip typical header label
            if cell.lower() == "cellid":
                continue
            s.add(cell)
    return s


def read_spots(path: Path):
    s = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            spot = row[0].strip()
            if not spot:
                continue
            # skip typical header label
            if spot.lower() == "spotid":
                continue
            s.add(spot)
    return s


def main(argv=None):

    dataset = Path("/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP")

    scGenes = read_genes(dataset / "scData_Genes.csv")
    stGenes = read_genes(dataset / "stData_Genes.csv")

    # scCells = read_cells(dataset / "scData_Cells.csv")
    # stSpots = read_spots(dataset / "stData_Spots.csv")
    shared = scGenes & stGenes

    # print("sc Cell", len(scCells))
    # print("st Spots", len(stSpots))
    print("sc Genes", len(scGenes))
    print("st genes", len(stGenes))
    print("Shared genes", len(shared))

    print(shared)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

