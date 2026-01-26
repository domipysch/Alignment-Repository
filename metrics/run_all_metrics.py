import argparse
from pathlib import Path
import logging
import sys
from .metrics_o1 import main as main1
from .metrics_o2 import main as main2
from .metrics_o2_permutation_test import main as main2permutation
from .metrics_o4_permutation_test import main as main4permutation
from .metrics_o4 import main as main4
logger = logging.getLogger(__name__)


def main(dataset: Path, result: Path, metrics: Path, run_permutation_tests: bool = False):

    # Run metrics computations
    main1(dataset, result, metrics)
    main2(dataset, result, metrics)
    main4(dataset, result, metrics)

    # Run permutation tests
    if run_permutation_tests:
        main2permutation(dataset, result, metrics)
        main4permutation(dataset, result, metrics)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run all metrics on a result file")
    parser.add_argument('-d', '--dataset', type=Path, help='Path to dataset folder')
    parser.add_argument('-r', '--result', type=Path, help='Path to result file')
    parser.add_argument('-m', '--metrics', type=Path, help='Path to output metric folder')

    args = parser.parse_args()

    logger.info("Starting metrics computation for:")
    logger.info("Dataset path: %s", args.dataset)
    logger.info("Result file path: %s", args.result)
    logger.info("Metrics output folder: %s", args.metrics)

    main(args.dataset, args.result, args.metrics)
