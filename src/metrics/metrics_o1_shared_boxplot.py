from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json


def create_marker_nonmarker_boxplot(
    json_paths, output_path, pair_labels=None, title=None, eps=1e-8
):
    """
    Erwartet genau 4 JSON-Dateien (je Datei: {"marker_norms": [...], "non_marker_norms": [...]})
    Erzeugt einen Boxplot mit 8 Boxen: für jede Datei (Paar) Marker then Non‑Marker.
    Werte werden als log10(norm + eps) geplottet.
    """
    if len(json_paths) != 4:
        raise ValueError("Es werden genau 4 JSON-Dateien (vier Paare) erwartet.")
    data = []
    for p in json_paths:
        with open(p, "r") as f:
            d = json.load(f)
        m = np.asarray(d.get("marker_norms", []), dtype=float)
        nm = np.asarray(d.get("non_marker_norms", []), dtype=float)
        data.append(np.log10(m + eps) if m.size > 0 else np.array([]))
        data.append(np.log10(nm + eps) if nm.size > 0 else np.array([]))

    # Positionen: 1..8
    n_pairs = 4
    positions = np.arange(1, 2 * n_pairs + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data, positions=positions, widths=0.6, patch_artist=True, showfliers=False
    )

    # Farben: Marker (even indices 0,2,..) vs Non-Marker (odd)
    marker_color = "#1f77b4"
    non_marker_color = "#ff7f0e"
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(marker_color if i % 2 == 0 else non_marker_color)
        box.set_alpha(0.7)
    for median in bp.get("medians", []):
        median.set_color("black")

    # x-ticks in die Mitte jeder Pair (1.5, 3.5, 5.5, 7.5)
    centers = np.arange(1.5, 2 * n_pairs + 1, 2)
    if pair_labels is None:
        pair_labels = ["scRNA Input X", "Tangram", "DOT", "TACCO"]
    ax.set_xticks(centers)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right")

    ax.set_ylabel("log10(norm)")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legende mit farbigen Rechtecken
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=marker_color, label="Marker"),
        Patch(facecolor=non_marker_color, label="Non-Marker"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outp, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    paths = [
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_HumanBreastCancer_CID4465/metrics_cell/scRNA_Metrics.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_HumanBreastCancer_CID4465/metrics_cell/tangram/z1/z1_metrics.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_HumanBreastCancer_CID4465/metrics_cell/dot/z1/z1_metrics.json",
        "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_HumanBreastCancer_CID4465/metrics_cell/tacco/z1/z1_metrics.json",
    ]

    # Einfach Labels aus Dateinamen (vier Paare)
    pair_labels = ["scRNA Input X", "Tangram", "DOT", "TACCO"]

    out_path = "/Users/domi/Dev/MPA_Workspace/MPA_DATA/01_HumanBreastCancer_CID4465/metrics_overall/z1-overall-cell.jpg"

    create_marker_nonmarker_boxplot(
        paths,
        out_path,
        pair_labels=pair_labels,
        title="Euclidean L2 Norm of Marker vs Non-Marker Gene Expressions",
    )
