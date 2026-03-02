import logging
import argparse
from pathlib import Path
import os
from .run_tacco import tacco_align_data
from ..metrics import run_all_shared_boxplots
from ..metrics import run_all_metrics

if __name__ == "__main__":
    """
    Run TACCO alignment on a prepared dataset at given folder under 6 different settings:
    (Probabilistic vs Deterministic mapping) x (Individual cells vs Major cell types vs Minor cell types)
    """
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run TACCO alignment on a dataset folder"
    )
    parser.add_argument("-d", "--dataset", type=str, help="Path to dataset folder")
    parser.add_argument(
        "-o", "--output_folder", type=str, help="Path where to store result"
    )
    parser.add_argument(
        "-m", "--metrics_folder", type=str, help="Path where to store metrics"
    )
    args = parser.parse_args()

    dataset_folder = Path(args.dataset)
    if not dataset_folder.exists():
        logging.error(f"Dataset folder does not exist: {dataset_folder}")
        exit(1)

    # If output folder does not exist, create it
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output folder: {output_folder}")

    # If output folder not empty, error
    if any(output_folder.iterdir()):
        logging.error(
            f"Output folder is not empty: {output_folder}. Please provide an empty folder."
        )
        exit(1)

    # If metrics folder does not exist, create it
    metrics_folder = Path(args.metrics_folder)
    if not metrics_folder.exists():
        metrics_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created metrics folder: {metrics_folder}")

    # If metrics folder not empty, error
    if any(metrics_folder.iterdir()):
        logging.error(
            f"metrics_folder is not empty: {metrics_folder}. Please provide an empty folder."
        )
        exit(1)

    logging.info("Run 1/6: Prob, individual cells")
    output_path = os.path.join(args.output_folder, "prob_cells_GEP.csv")
    predicted_gep = tacco_align_data(
        args.dataset,
        deterministic_mapping=False,
        cell_type_key="cellID",
        output_path=output_path,
    )
    run_all_metrics.main(
        dataset_folder,
        metrics_folder / "prob_cells",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 2/6: Prob, Cell type major")
    predicted_gep = tacco_align_data(
        args.dataset,
        deterministic_mapping=False,
        cell_type_key="cellType",
        output_path=os.path.join(args.output_folder, "prob_celltype_major_GEP.csv"),
    )
    run_all_metrics.main(
        dataset_folder,
        metrics_folder / "prob_celltype_major",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 3/6: Prob, Cell type minor")
    predicted_gep = tacco_align_data(
        args.dataset,
        deterministic_mapping=False,
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.output_folder, "prob_celltype_minor_GEP.csv"),
    )
    run_all_metrics.main(
        dataset_folder,
        metrics_folder / "prob_celltype_minor",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 4/6: Det, individual cells")
    predicted_gep = tacco_align_data(
        args.dataset,
        deterministic_mapping=True,
        cell_type_key="cellID",
        output_path=os.path.join(args.output_folder, "det_cells_GEP.csv"),
    )
    run_all_metrics.main(
        dataset_folder,
        metrics_folder / "det_cells",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 5/6: Det, Cell type major")
    predicted_gep = tacco_align_data(
        args.dataset,
        deterministic_mapping=True,
        cell_type_key="cellType",
        output_path=os.path.join(args.output_folder, "det_celltype_major_GEP.csv"),
    )
    run_all_metrics.main(
        dataset_folder,
        metrics_folder / "det_celltype_major",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 6/6: Det, Cell type minor")
    predicted_gep = tacco_align_data(
        args.dataset,
        deterministic_mapping=True,
        cell_type_key="cellTypeMinor",
        output_path=os.path.join(args.output_folder, "det_celltype_minor_GEP.csv"),
    )
    run_all_metrics.main(
        dataset_folder,
        metrics_folder / "det_celltype_minor",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    # Create shared boxplots
    metric_folder_shared = metrics_folder / "shared"
    metric_folder_shared.mkdir(parents=True, exist_ok=True)
    folders = [
        "det_cells",
        "det_celltype_major",
        "det_celltype_minor",
        "prob_cells",
        "prob_celltype_major",
        "prob_celltype_minor",
    ]
    # Run shared metrics
    run_all_shared_boxplots.main(
        [metrics_folder / fol for fol in folders],
        folders,
        metric_folder_shared,
    )
