import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
import yaml
import itertools
import copy
import time
import csv
import traceback
import logging
import alternative_idea.main as alternative_idea
import metrics.run_all_metrics as run_all_metrics
import metrics.run_all_shared_boxplots as run_all_shared_boxplots
import metrics.run_all_permutation_boxplots as run_all_permutation_boxplots
logger = logging.getLogger(__name__)


def create_shared_boxplots(ids: list[int], metrics_folder: Path, output_folder: Path, run_permutation_tests: bool = False):

    # Run shared metrics
    run_all_shared_boxplots.main(
        [metrics_folder / f"{id}" for id in ids],
        list(map(str, ids)),
        output_folder,
    )

    # Run shared permutation test boxplots
    if run_permutation_tests:
        run_all_permutation_boxplots.main(
            [metrics_folder / f"{id}" for id in ids],
            list(map(str, ids)),
            output_folder,
        )


def run_config(dataset: Path, run_config_path: Path, save_result_path: Optional[Path], save_mapping_path: Optional[Path], metrics_folder: Path, run_permutation_tests: bool = False):
    # Determine verbose flag from current logger level
    verbose_flag = logger.getEffectiveLevel() == logging.DEBUG

    # Run alignment (G x S)
    predicted_gep = alternative_idea.main(
        dataset,
        run_config_path,
        output_path=save_result_path,
        mapping_output_path=save_mapping_path,
        verbose_logging=verbose_flag
    )

    # Run individual metrics
    run_all_metrics.main(
        dataset,
        metrics_folder,
        result_gep=predicted_gep,
        run_permutation_tests=run_permutation_tests
    )


def main(dataset: Path, experiment_config: Path, result_folder: Path, metric_folder: Path, save_result: bool = False, run_permutation_tests: bool = False):

    if not experiment_config.exists():
        raise FileNotFoundError(f"experiment_config not found: {experiment_config}")
    if not dataset.exists():
        raise FileNotFoundError(f"dataset folder not found: {dataset}")

    # Load experiment config
    with open(experiment_config, "r") as f:
        base_cfg = yaml.safe_load(f) or {}
    if not isinstance(base_cfg, dict):
        raise ValueError("Top-level experiment_config must be a mapping/dict.")

    # Helper: collect leaf paths -> list of values (lists become value lists, scalars become singleton list)
    def collect_leaves(node, path=()):
        """
        Traverses `node` (which may be nested dicts/lists/scalars) and returns a list of
        (path_tuple, values_list) pairs.

        New behavior: if at some path the node is a list AND every element of that list is a dict,
        we treat the whole list as a set of alternative full-config dictionaries for that path.
        This allows specifying e.g. multiple complete `loss_weights` dicts as alternative configs.

        Examples handled:
        - scalar -> becomes [scalar]
        - list of scalars -> becomes that list
        - list of dicts -> treated as a leaf; values_list equals the list of dicts
        - dict -> recurse into keys
        """
        leaves = []
        # If it's a list at this path
        if isinstance(node, list):
            # empty list -> keep as-is (will be validated later)
            if len(node) == 0:
                leaves.append((path, []))
                return leaves
            # if all items are dicts, treat the whole list as an atomic set of dict-options
            if all(isinstance(item, dict) for item in node):
                leaves.append((path, node))
                return leaves
            # otherwise treat it as a normal list of scalar options
            leaves.append((path, node))
            return leaves

        # If it's a dict, recurse into keys
        if isinstance(node, dict):
            for k, v in node.items():
                leaves.extend(collect_leaves(v, path + (k,)))
            return leaves

        # Otherwise scalar leaf
        vals = [node]
        leaves.append((path, vals))
        return leaves

    leaves = collect_leaves(base_cfg)

    # Ensure no leaf has empty list
    for path, vals in leaves:
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"Configuration entry {'.'.join(path)} must be a non-empty list or scalar.")

    # Prepare ordered lists for product
    paths = [p for p, v in leaves]
    lists = [v for p, v in leaves]

    total_runs = 1
    for v in lists:
        total_runs *= len(v)

    logger.info(f"Experiment config loaded from {experiment_config}. Total runs to execute: {total_runs}")

    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    if os.path.exists(metric_folder):
        shutil.rmtree(metric_folder)
    result_folder.mkdir(parents=True, exist_ok=False)
    metric_folder.mkdir(parents=True, exist_ok=False)

    # Prepare summary CSV
    summary_path = result_folder / "summary.csv"
    write_header = not summary_path.exists()

    # Iterate over grid
    combo_iter = itertools.product(*lists)

    # Function to set a value in nested dict by path
    def set_in_dict(d: dict, path: tuple, value):
        cur = d
        for key in path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[path[-1]] = value

    run_id = 0
    with open(summary_path, "a", newline='') as summary_file:

        writer = csv.writer(summary_file)
        if write_header:
            writer.writerow(["id", "config_path", "output_path", "status", "duration_seconds", "error_message"])

        for combo in combo_iter:
            # Build run-specific config
            cfg_copy = copy.deepcopy(base_cfg)
            for path, val in zip(paths, combo):
                set_in_dict(cfg_copy, path, val)

            run_dir = result_folder / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            run_config_path = run_dir / "config.yml"
            with open(run_config_path, "w") as cf:
                yaml.safe_dump(cfg_copy, cf, sort_keys=False)

            result_path = run_dir / "result_GEP.csv"

            metric_dir = metric_folder / str(run_id)
            metric_dir.mkdir(parents=True, exist_ok=False)

            start = time.time()
            try:
                logger.info(f"Starting run {run_id}/{total_runs - 1} -> writing to {run_dir}")
                run_config(
                    dataset,
                    run_config_path,
                    (run_dir / "gep.csv") if save_result else None,
                    (run_dir / "mapping.csv") if save_result else None,
                    metric_dir,
                    run_permutation_tests=run_permutation_tests
                )
                duration = time.time() - start
                writer.writerow([run_id, str(run_config_path), str(result_path), "ok", f"{duration:.3f}", ""])
                logger.info(f"Run {run_id} completed in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start
                tb = traceback.format_exc()
                logger.error(f"Run {run_id} failed after {duration:.2f}s: {e}\n{tb}")
                writer.writerow([run_id, str(run_config_path), str(result_path), "error", f"{duration:.3f}", str(e)])
                # Stop on first error as requested
                raise

            run_id += 1

    # Create shared boxplots
    metric_folder_shared = metric_folder / "shared"
    metric_folder_shared.mkdir(parents=True, exist_ok=False)
    create_shared_boxplots(list(range(run_id)), metric_folder, metric_folder_shared, run_permutation_tests=run_permutation_tests)


if __name__ == "__main__":

    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run AlternativeIdea alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', type=Path, help='Path to dataset folder')
    parser.add_argument('-c', '--experiment_config', type=Path, help='Path to config.yaml')
    parser.add_argument('-o', '--result_folder', type=Path, help='Path where to store results to')
    parser.add_argument('-s', '--save_result', action='store_true', help='Whether to save the predicte Z prime to disk')
    parser.add_argument('-m', '--metric_folder', type=Path, help='Path where to store metrics to')
    parser.add_argument('--logging', dest='logging', choices=['normal', 'verbose'], default='normal',
                        help="Logging verbosity. Use 'verbose' for more logs.")
    parser.add_argument('--run_permutation_tests', dest='run_permutation_tests', action='store_true', help="Whether to run permutation tests or not.")
    args = parser.parse_args()

    # 2. Configure logging based on argument
    level = logging.DEBUG if args.logging == "verbose" else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(level)

    # 3. Run
    main(
        args.dataset,
        args.experiment_config,
        args.result_folder,
        args.metric_folder,
        save_result=args.save_result,
        run_permutation_tests=args.run_permutation_tests
    )
