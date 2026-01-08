import numpy as np
import matplotlib.pyplot as plt
import json


def compute_medians(paths_to_jsons: list[str], labels: list[str]) -> dict:
    """
    Compute the median of numeric values contained in each JSON file.
    """
    medians = {}
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


def main(paths_to_jsons: list[str], labls: list[str], title: str, output_path: str = None):
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
    plt.ylabel("Custom locality metric", fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(rotation=30, fontsize=xtick_fontsize)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":

    paths = [
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_cell/tangram/z4/knn/cossim.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_cellTypeMinor/tangram/z4/knn/cossim.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_cellType/tangram/z4/knn/cossim.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_cell/tangram_non-det/z4/knn/cossim.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_cellTypeMinor/tangram_non-det/z4/knn/cossim.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_cellType/tangram_non-det/z4/knn/cossim.json",
    ]

    labels = [
        "Cell - det.",
        "Minor cell state - det.",
        "Major cell state - det.",
        "Cell - prob.",
        "Minor cell state - prob.",
        "Major cell state - prob.",
    ]

    print(json.dumps(compute_medians(paths, labels), indent=4))

    out_path = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/03_MouseSSP/metrics_overall/z4-cossim-tangram.jpg"
    main(
        paths,
        labels,
        "LSO Data - TACCO",
        output_path=out_path
    )
