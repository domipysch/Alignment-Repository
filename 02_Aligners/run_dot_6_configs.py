import logging
import argparse
from pathlib import Path
import os
from run_dot import dot_align_data


if __name__ == "__main__":
    """
    Run DOT alignment on a prepared dataset at given folder.
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run DOT alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', type=str, help='Path to dataset folder')
    parser.add_argument('-o', '--output_folder', type=str, help='Path where to store result')
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
    dot_align_data(
        args.dataset,
        "HSO",
        "probabilistic-mapping",
        cell_type_key="cellID",
        output_path=os.path.join(args.output_folder, "prob_cells_GEP.csv"),
    )

    logging.info("Run 2/6: Prob, Cell type major")
    dot_align_data(
        args.dataset,
        "HSO",
        "probabilistic-mapping",
        cell_type_key="cellType",
        output_path=os.path.join(args.output_folder, "prob_celltype_major_GEP.csv"),
    )

    logging.info("Run 3/6: Prob, Cell type minor")
    dot_align_data(
        args.dataset,
        "HSO",
        "probabilistic-mapping",
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.output_folder, "prob_celltype_minor_GEP.csv"),
    )

    logging.info("Run 4/6: Det, individual cells")
    dot_align_data(
        args.dataset,
        "HSO",
        "deterministic-mapping",
        cell_type_key="cellID",
        output_path=os.path.join(args.output_folder, "det_cells_GEP.csv"),
    )

    logging.info("Run 5/6: Det, Cell type major")
    dot_align_data(
        args.dataset,
        "HSO",
        "deterministic-mapping",
        cell_type_key="cellType",
        output_path=os.path.join(args.output_folder, "det_celltype_major_GEP.csv"),
    )

    logging.info("Run 6/6: Det, Cell type minor")
    dot_align_data(
        args.dataset,
        "HSO",
        "deterministic-mapping",
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.output_folder, "det_celltype_minor_GEP.csv"),
    )

