import argparse
from pathlib import Path
import logging
import sys
from metrics_o1 import main as main1
from metrics_o2 import main as main2
from metrics_o2_permutation_test import main as main2permutation
from metrics_o4_permutation_test import main as main4permutation
from metrics_o4 import main as main4
logger = logging.getLogger(__name__)


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

    # Run metrics computations
    main1(args.dataset, args.result, args.metrics)
    main2(args.dataset, args.result, args.metrics)
    main4(args.dataset, args.result, args.metrics)

    # Run permutation tests
    main2permutation(args.dataset, args.result, args.metrics)
    main4permutation(args.dataset, args.result, args.metrics)

