import logging
import argparse
from pathlib import Path

from run_tangram import tangram_align_data


if __name__ == "__main__":
    """
    Run Tangram alignment on a prepared dataset at given folder.
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run Tangram alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', type=str, help='Path to dataset folder')
    parser.add_argument('-o', '--output_folder', type=str, help='Path where to store result')
    parser.add_argument('-nal', '--normalize_and_log', action='store_true', help='Whether to normalize and log input data beforehand')
    args = parser.parse_args()

    # If output folder does not exist, create it
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output folder: {output_folder}")

    # If output folder not empty, error
    if any(output_folder.iterdir()):
        logging.error(f"Output folder is not empty: {output_folder}. Please provide an empty folder.")
        exit(1)

    logging.info("Run 1/6: Prob, individual cells")
    tangram_align_data(
        args.dataset,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=False,
        cell_type_key="cellType",
        output_path=Path(args.output_folder) / "prob_cells_GEP.csv",
    )

    logging.info("Run 2/6: Prob, Major cell types")
    tangram_align_data(
        args.dataset,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellType",
        output_path=Path(args.output_folder) / "prob_celltype_major_GEP.csv",
    )

    logging.info("Run 3/6: Prob, Minor cell types")
    tangram_align_data(
        args.dataset,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellTypeMinor",
        output_path=Path(args.output_folder) / "prob_celltype_minor_GEP.csv",
    )

    logging.info("Run 4/6: Det, individual cells")
    tangram_align_data(
        args.dataset,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=False,
        cell_type_key="cellType",
        output_path=Path(args.output_folder) / "det_cells_GEP.csv",
    )

    logging.info("Run 5/6: Det, Major cell types")
    tangram_align_data(
        args.dataset,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellType",
        output_path=Path(args.output_folder) / "det_celltype_major_GEP.csv",
    )

    logging.info("Run 6/6: Det, Minor cell types")
    tangram_align_data(
        args.dataset,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellTypeMinor",
        output_path=Path(args.output_folder) / "det_celltype_minor_GEP.csv",
    )
