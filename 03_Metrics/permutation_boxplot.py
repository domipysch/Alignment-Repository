import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})


def load(path):
    d = json.loads(Path(path).read_text())
    return list(map(float, d["T_permuted"])), float(d["T_original"])


def main(input: Path, output: Path):
    """
    Create boxplot for a single permutation test (just one distribution, one specific method + config).

    Args:
        input:
        output:

    Returns:

    """
    input = Path(input)
    if not input.exists():
        print(f"Datei nicht gefunden: {input}", file=sys.stderr)
        return
    try:
        perm, orig = load(input)
    except Exception as e:
        print(f"Fehler beim Lesen {input}: {e}", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(4, 1.2), 5))
    ax.boxplot([perm], tick_labels=[input.stem], patch_artist=True, medianprops={"color":"black"})
    ax.scatter([1], [orig], color="red", marker="D", zorder=5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)


def combined_boxplot(json_paths, output: Path, labels=None, title=""):
    """
    Create one boxplot from multiple permutation test JSON files.
    (e.g. per method: all 6 configs)
    Args:
        json_paths:
        output:
        labels:
        title:

    Returns:

    """
    json_paths = [Path(p) for p in json_paths]
    boxes = []
    originals = []
    names = []

    for i, p in enumerate(json_paths):
        if not p.exists():
            print(f"Datei nicht gefunden, übersprungen: {p}", file=sys.stderr)
            continue
        try:
            perm, orig = load(p)
        except Exception as e:
            print(f"Fehler beim Lesen {p}, übersprungen: {e}", file=sys.stderr)
            continue
        boxes.append(list(map(float, perm)))
        originals.append(float(orig))
        if labels and i < len(labels):
            names.append(labels[i])
        else:
            names.append(p.stem)

    if not boxes:
        print("Keine gültigen JSON-Dateien gefunden.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(boxes)), 5))
    ax.boxplot(boxes, patch_artist=True, medianprops={"color": "black"})
    xs = list(range(1, len(boxes) + 1))
    ax.scatter(xs, originals, color="red", marker="D", zorder=5)
    ax.set_xticks(xs)
    ax.set_title(title)
    # explizite Fontgrößen für xticks und ylabel
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("T = Sum of custom locality metric", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create boxplots for permutation test results (single JSON per call)")
    parser.add_argument('-d', '--dataset', type=str, help='Path to dataset folder', required=True)
    args = parser.parse_args()

    # methods = ["tangram", "tangram_non-det", "dot", "dot_non-det", "tacco", "tacco_non-det"]
    methods = ["tangram", "tangram_non-det"]
    result_folders = ["results_cell", "results_cellType", "results_cellTypeMinor"]
    metric_folders = ["metrics_cell", "metrics_cellType", "metrics_cellTypeMinor"]

    combined_boxplot(
        [
            Path(args.dataset) / "metrics_cell" / "tangram" / "z3" / "permutation_test" / "permutation_test_per_gene.json",
            Path(args.dataset) / "metrics_cellTypeMinor" / "tangram" / "z3" / "permutation_test" / "permutation_test_per_gene.json",
            Path(args.dataset) / "metrics_cellType" / "tangram" / "z3" / "permutation_test" / "permutation_test_per_gene.json",
            Path(args.dataset) / "metrics_cell" / "tangram_non-det" / "z3" / "permutation_test" / "permutation_test_per_gene.json",
            Path(args.dataset) / "metrics_cellTypeMinor" / "tangram_non-det" / "z3" / "permutation_test" / "permutation_test_per_gene.json",
            Path(args.dataset) / "metrics_cellType" / "tangram_non-det" / "z3" / "permutation_test" / "permutation_test_per_gene.json",
        ],
        Path(args.dataset) / "metrics_overall" / "z3_permutation_test-tangram.png",
        labels=[
            "Cell - det.",
            "Minor cell state - det.",
            "Major cell state - det.",
            "Cell - prob.",
            "Minor cell state - prob.",
            "Major cell state - prob.",
        ],
        title="Tangram"
    )

    # Z3
    # for method in methods:
    #     for result_folder, metric_folder in zip(result_folders, metric_folders):
    #
    #         base = Path(args.dataset) / metric_folder / method / "z3" / "permutation_test"
    #         # jeweils einzeln aufrufen: per_gene
    #         gene_file = base / "permutation_test_per_gene.json"
    #         out_gene = Path(args.dataset) / metric_folder / method / "z3" / "permutation_test" / "permutation_boxplot_per_gene.png"
    #         main(gene_file, out_gene)
    #         # per_spot
    #         spot_file = base / "permutation_test_per_spot.json"
    #         out_spot = Path(args.dataset) / metric_folder / method / "z3" / "permutation_test" / "permutation_boxplot_per_spot.png"
    #         main(spot_file, out_spot)

    # Z4
    # for method in methods:
    #     for result_folder, metric_folder in zip(result_folders, metric_folders):
    #
    #         base = Path(args.dataset) / metric_folder / method / "z4" / "knn"
    #         # jeweils einzeln aufrufen: per_gene
    #         in_file = base / "permutation_test.json"
    #         out_file = Path(args.dataset) / metric_folder / method / "z4" / "knn" / "permutation_test.png"
    #         main(in_file, out_file)
