import argparse
from metrics_o1 import main as main1
from metrics_o2 import main as main2
from metrics_o4 import main as main4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all metrics on a result file")
    parser.add_argument('-d', '--dataset', type=str, help='Path to dataset folder')
    args = parser.parse_args()

    methods = ["tangram", "tangram_non-det", "dot", "dot_non-det", "tacco", "tacco_non-det"]
    result_folders = ["results_cell", "results_cellType", "results_cellTypeMinor"]
    metric_folders = ["metrics_cell", "metrics_cellType", "metrics_cellTypeMinor"]

    for method in methods:
        for result_folder, metric_folder in zip(result_folders, metric_folders):
            main1(args.dataset, result_folder, metric_folder, method)
            main2(args.dataset, result_folder, metric_folder, method)
            main4(args.dataset, result_folder, metric_folder, method)
