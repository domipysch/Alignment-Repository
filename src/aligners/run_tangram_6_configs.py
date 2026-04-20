import logging
import argparse
from pathlib import Path
from ..metrics import run_all_shared_boxplots, run_all_permutation_boxplots
from ..metrics import run_all_metrics
from .run_tangram import tangram_align_data

if __name__ == "__main__":
    """
    Run Tangram alignment on a prepared dataset at given folder with 6 different settings:
    (Probabilistic vs Deterministic mapping) x (Individual cells vs Major cell types vs Minor cell types)
    """

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Tangram alignment on a dataset folder"
    )
    parser.add_argument(
        "--scdata", type=str, required=True, help="Full path to sc.h5ad"
    )
    parser.add_argument(
        "--stdata", type=str, required=True, help="Full path to st.h5ad"
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, help="Path where to store result"
    )
    parser.add_argument(
        "-m", "--metrics_folder", type=str, help="Path where to store metrics"
    )
    parser.add_argument(
        "-nal",
        "--normalize_and_log",
        action="store_true",
        help="Whether to normalize and log input data beforehand",
    )
    args = parser.parse_args()

    # If output folder does not exist, create it
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output folder: {output_folder}")

    # If metrics folder does not exist, create it
    metrics_folder = Path(args.metrics_folder)
    if not metrics_folder.exists():
        metrics_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created metrics folder: {metrics_folder}")

    logging.info("Run 1/6: Prob, individual cells")
    output_path = Path(args.output_folder) / "prob_cells_GEP.h5ad"
    predicted_gep = tangram_align_data(
        args.scdata,
        args.stdata,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=False,
        cell_type_key="cellType",
        output_path=output_path,
    )
    # predicted_gep = ad.read_h5ad(output_path)  # If already computed
    run_all_metrics.main(
        args.scdata,
        args.stdata,
        metrics_folder / "prob_cells",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 2/6: Prob, Major cell types")
    output_path = Path(args.output_folder) / "prob_celltype_major_GEP.h5ad"
    predicted_gep = tangram_align_data(
        args.scdata,
        args.stdata,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=False,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellType",
        output_path=output_path,
    )
    # predicted_gep = ad.read_h5ad(output_path) # If already computed
    run_all_metrics.main(
        args.scdata,
        args.stdata,
        metrics_folder / "prob_celltype_major",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    # logging.info("Run 3/6: Prob, Minor cell types")
    # output_path = Path(args.output_folder) / "prob_celltype_minor_GEP.h5ad"
    # predicted_gep = tangram_align_data(
    #     args.scdata,
    #     args.stdata,
    #     normalize_and_log=args.normalize_and_log,
    #     deterministic_mapping=False,
    #     compute_marker_genes=False,
    #     map_clusters=True,
    #     cell_type_key="cellTypeMinor",
    #     output_path=output_path,
    # )
    # predicted_gep = ad.read_h5ad(output_path)  # If already computed
    # run_all_metrics.main(
    #     dataset_folder,
    #     metrics_folder / "prob_celltype_minor",
    #     result_gep=predicted_gep,
    #     run_permutation_tests=False,
    # )

    logging.info("Run 4/6: Det, individual cells")
    output_path = Path(args.output_folder) / "det_cells_GEP.h5ad"
    predicted_gep = tangram_align_data(
        args.scdata,
        args.stdata,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=False,
        cell_type_key="cellType",
        output_path=output_path,
    )
    # predicted_gep = ad.read_h5ad(output_path)  # If already computed
    run_all_metrics.main(
        args.scdata,
        args.stdata,
        metrics_folder / "det_cells",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    logging.info("Run 5/6: Det, Major cell types")
    output_path = Path(args.output_folder) / "det_celltype_major_GEP.h5ad"
    predicted_gep = tangram_align_data(
        args.scdata,
        args.stdata,
        normalize_and_log=args.normalize_and_log,
        deterministic_mapping=True,
        compute_marker_genes=False,
        map_clusters=True,
        cell_type_key="cellType",
        output_path=output_path,
    )
    # predicted_gep = ad.read_h5ad(output_path)  # If already computed
    run_all_metrics.main(
        args.scdata,
        args.stdata,
        metrics_folder / "det_celltype_major",
        result_gep=predicted_gep,
        run_permutation_tests=False,
    )

    # logging.info("Run 6/6: Det, Minor cell types")
    # output_path = Path(args.output_folder) / "det_celltype_minor_GEP.h5ad"
    # predicted_gep = tangram_align_data(
    #     args.scdata,
    #     args.stdata,
    #     normalize_and_log=args.normalize_and_log,
    #     deterministic_mapping=True,
    #     compute_marker_genes=False,
    #     map_clusters=True,
    #     cell_type_key="cellTypeMinor",
    #     output_path=output_path,
    # )
    # predicted_gep = ad.read_h5ad(output_path)  # If already computed
    # run_all_metrics.main(
    #     dataset_folder,
    #     metrics_folder / "det_celltype_minor",
    #     result_gep=predicted_gep,
    #     run_permutation_tests=False,
    # )

    # Create shared boxplots
    metric_folder_shared = metrics_folder / "shared"
    metric_folder_shared.mkdir(parents=True, exist_ok=True)
    folders = [
        "prob_cells",
        "det_cells",
        # "prob_celltype_minor",
        # "det_celltype_minor",
        "prob_celltype_major",
        "det_celltype_major",
    ]
    labels = [
        "Cell - prob.",
        "Cell - det.",
        # "Minor cell state - prob.",
        # "Minor cell state - det.",
        "Major cell state - prob.",
        "Major cell state - det.",
    ]
    # Run shared metrics
    run_all_shared_boxplots.main(
        [metrics_folder / fol for fol in folders],
        labels,
        metric_folder_shared,
    )
    run_all_permutation_boxplots.main(
        [metrics_folder / fol for fol in folders],
        labels,
        metric_folder_shared,
    )
