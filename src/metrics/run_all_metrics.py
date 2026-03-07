import argparse
from pathlib import Path
import logging
import sys
from anndata import AnnData
from ..utils.io import csv_to_anndata
from .metrics_o1 import main as main1
from .metrics_o2 import main as main2
from .metrics_o4 import main as main4
from .metrics_o2_permutation_test import main as main2permutation
from .metrics_o4_permutation_test import main as main4permutation

logger = logging.getLogger(__name__)


def main(
    dataset: Path,
    metrics: Path,
    result_gep: AnnData,
    run_permutation_tests: bool = False,
):
    """

    Args:
        dataset:
        result_gep: Predicted Z' (G x S)
        metrics:
        run_permutation_tests:

    Returns:

    """
    # Run metrics computations
    main1(dataset, result_gep, metrics)
    main2(dataset, result_gep, metrics)
    main4(dataset, result_gep, metrics)

    # Run permutation tests
    if run_permutation_tests:
        main2permutation(dataset, result_gep, metrics)
        main4permutation(dataset, result_gep, metrics)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run all metrics on a result file")
    parser.add_argument("-d", "--dataset", type=Path, help="Path to dataset folder")
    parser.add_argument("-r", "--result", type=Path, help="Path to result file")
    parser.add_argument(
        "-m", "--metrics", type=Path, help="Path to output metric folder"
    )

    args = parser.parse_args()

    logger.info("Starting metrics computation for:")
    logger.info("Dataset path: %s", args.dataset)
    logger.info("Result file path: %s", args.result)
    logger.info("Metrics output folder: %s", args.metrics)

    result_gep = csv_to_anndata(args.result, transpose=False)
    main(args.dataset, args.metrics, result_gep, run_permutation_tests=False)
