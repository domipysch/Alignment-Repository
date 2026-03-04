import subprocess
import argparse
import os
import shutil
from pathlib import Path

R_SCRIPT = os.path.join(os.path.dirname(__file__), "run_dot.R")


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    fallback = "/opt/miniconda3/envs/dot_env/bin/Rscript"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        "Rscript not found. Activate the dot_env conda environment or add it to PATH."
    )


def dot_align_data(
    dataset_folder: str,
    mode: str,
    mapping_mode: str,
    cell_type_key: str,
    output_path: Path,
):
    """
    Run DOT alignment by calling run_dot.R via Rscript.

    Args:
        dataset_folder: Path to dataset folder containing scData_GEP.csv, stData_GEP.csv, etc.
        mode: "LSO" or "HSO"
        mapping_mode: "deterministic-mapping" or "probabilistic-mapping"
        cell_type_key: Column name in scData_Cells.csv containing cell type labels
        output_path: Output path for the resulting GEP CSV
    """
    cmd = [
        _find_rscript(),
        R_SCRIPT,
        dataset_folder,
        mode,
        mapping_mode,
        cell_type_key,
        output_path,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOT alignment via run_dot.R")
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset folder")
    parser.add_argument(
        "-m", "--mode", default="HSO", choices=["LSO", "HSO"], help="Resolution mode"
    )
    parser.add_argument(
        "--mapping",
        default="deterministic-mapping",
        choices=["deterministic-mapping", "probabilistic-mapping"],
        help="Mapping mode",
    )
    parser.add_argument(
        "-k",
        "--cell-type-key",
        default="cellID",
        help="Column in scData_Cells.csv with cell type labels",
    )
    parser.add_argument("-o", "--output", required=True, help="Output path for GEP CSV")
    args = parser.parse_args()

    dot_align_data(
        args.dataset, args.mode, args.mapping, args.cell_type_key, args.output
    )
