from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
import sys
import argparse

logger = logging.getLogger(__name__)


def compute_medians(paths_to_jsons: list[Path], labels: list[str]) -> dict:
    """
    Compute the median of numeric values contained in each JSON file.
    """
    medians: dict[str, float | None] = {}
    for path, label in zip(paths_to_jsons, labels):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support dict (key->value) or list
            if isinstance(data, dict):
                raw_values = list(data.values())
            elif isinstance(data, list):
                raw_values = data
            else:
                medians[label] = None
                continue

            nums = []
            for v in raw_values:
                if v is None:
                    continue
                try:
                    nums.append(float(v))
                except Exception:
                    continue

            if not nums:
                medians[label] = None
            else:
                medians[label] = float(np.median(nums))
        except Exception:
            medians[label] = None
    return medians


def create_shared_boxplot(
    paths_to_jsons: list[Path],
    labls: list[str],
    title: str,
    ylabel: str,
    output_path: Path | None = None,
):
    """
    Create a combined boxplot from multiple JSON files.
    Values are extracted from each file and plotted as side-by-side boxplots.
    If output_path is provided the figure is saved, otherwise it is shown interactively.
    """
    # Collect data from all JSON files
    all_data = []
    for path in paths_to_jsons:
        with open(path, "r") as f:
            data = json.load(f)
            values = list(data.values())
            all_data.append(values)

    # --- Modified/additional lines: increase font sizes ---
    # Increase global default font size
    plt.rcParams.update({"font.size": 14})
    # Optional: separate sizes for title and axis labels
    title_fontsize = 20
    label_fontsize = 18
    xtick_fontsize = 18
    # --------------------------------------------------------

    # Create boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_data, tick_labels=labls)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(rotation=30, fontsize=xtick_fontsize)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def main(metrics_paths: list[Path], labels: list[str], output_folder: Path):

    # Create shared boxplot for o2, genewise
    create_shared_boxplot(
        [path / "o2" / "boxplots_per_gene" / "cossim.json" for path in metrics_paths],
        labels,
        "o2 across runs",
        "Genewise cosine similarity",
        output_path=output_folder / "o2_genewise_overall.png",
    )

    # Create shared boxplot for o2, spotwise
    create_shared_boxplot(
        [path / "o2" / "boxplots_per_spot" / "cossim.json" for path in metrics_paths],
        labels,
        "o2 across runs",
        "Spotwise cosine similarity",
        output_path=output_folder / "o2_spotwise_overall.png",
    )

    # Create shared boxplot for o4
    create_shared_boxplot(
        [path / "o4" / "knn" / "cossim.json" for path in metrics_paths],
        labels,
        "o4 across runs",
        "Custom locality metric",
        output_path=output_folder / "o4_overall.png",
    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Create combined boxplots from multiple metric folders."
    )
    parser.add_argument(
        "-m", "--metrics", nargs="+", type=Path, help="Path to output metric folders"
    )
    parser.add_argument(
        "-l", "--labels", nargs="+", type=str, help="Label for each box"
    )
    parser.add_argument(
        "-o", "--output_folder", type=Path, help="Path to output folder"
    )
    args = parser.parse_args()

    logger.info("Create shared boxplots for:")
    logger.info("Metrics: %s", args.metrics)
    logger.info("Metrics output folder: %s", args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if len(args.metrics) != len(args.labels):
        raise ValueError("Anzahl der --paths muss gleich der Anzahl der --labels sein.")

    main(args.metrics, args.labels, args.output_folder)
