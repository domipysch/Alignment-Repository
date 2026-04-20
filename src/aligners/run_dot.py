import subprocess
import argparse
import os
import shutil
import logging
import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
from pathlib import Path

R_SCRIPT = os.path.join(os.path.dirname(__file__), "run_dot.R")

logger = logging.getLogger(__name__)


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
    sc_path: Path,
    st_path: Path,
    mode: str,
    mapping_mode: str,
    map_cell_types: bool,
    cell_type_key: str = "cellType",
    output_path: Path = None,
) -> AnnData:
    """
    Run DOT alignment by calling run_dot.R via Rscript.
    Saves the resulting GEP as h5ad to output_path and returns it.

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        mode: "LSO" or "HSO"
        mapping_mode: "deterministic-mapping" or "probabilistic-mapping"
        map_cell_types: If True, aggregate cells by cell_type_key before mapping.
                        If False, map individual cells (uses cellID as the annotation key).
        cell_type_key: obs column to use as annotation when map_cell_types=True.
        output_path: Output path for the resulting GEP h5ad (suffix forced to .h5ad)

    Returns:
        AnnData with obs=genes, var=spots (G x S layout)
    """
    annotation_key = cell_type_key if map_cell_types else "cellID"

    output_path = Path(output_path).with_suffix(".h5ad")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # R can only write CSV reliably; use a sidecar path for the R output
    csv_path = output_path.with_suffix(".csv")

    cmd = [
        _find_rscript(),
        R_SCRIPT,
        str(sc_path),
        str(st_path),
        mode,
        mapping_mode,
        annotation_key,
        str(csv_path),
    ]
    logger.info("Running DOT via R: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Convert R-written CSV (G x S, index col = gene names) to h5ad
    df = pd.read_csv(csv_path, header=0, index_col=0)
    X = np.asarray(df.values, dtype=np.float32)
    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    adata.write_h5ad(output_path)
    logger.info("Saved DOT GEP to %s", output_path)

    return adata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run DOT alignment via run_dot.R")
    parser.add_argument(
        "--scdata", type=Path, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=Path, required=True, help="Full path to st.h5ad"
    )
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
        "--map-cell-types",
        action="store_true",
        default=False,
        help="If set, aggregate cells by cell_type_key before mapping. Otherwise map individual cells.",
    )
    parser.add_argument(
        "-k",
        "--cell-type-key",
        default="cellType",
        help="obs column to use as annotation when --map-cell-types is set.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output path for GEP h5ad"
    )
    args = parser.parse_args()

    dot_align_data(
        args.scdata,
        args.stdata,
        args.mode,
        args.mapping,
        map_cell_types=args.map_cell_types,
        cell_type_key=args.cell_type_key,
        output_path=args.output,
    )
