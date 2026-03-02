from copy import copy
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from anndata import AnnData
from scipy import sparse
from .utils.utils import compute_basic_metrics_for_gene_groups
from .utils.dataset_query import get_shared_genes
logger = logging.getLogger(__name__)


def compute_metrics_scRNA(dataset_folder: Path) -> Dict[str, float]:
    """
    Load input scRNA GEP file as pandas DataFrame & compute basic metrics.
    """

    gep_path = dataset_folder / "scData_GEP.csv"
    if not gep_path.exists():
        raise FileNotFoundError(f"Ergebnisdatei nicht gefunden: {gep_path}")

    # Read dataframe
    df_gep = pd.read_csv(gep_path, header=0, index_col=0)
    sc_genes = set(df_gep.index)

    # Split result genes in marker and non-marker genes
    marker_genes = set(get_shared_genes(dataset_folder))
    marker_genes_sc = list(sc_genes.intersection(marker_genes))
    non_marker_genes_sc = list(sc_genes.difference(marker_genes))

    return compute_basic_metrics_for_gene_groups(
        df_gep,
        marker_genes_sc,
        non_marker_genes_sc,
        include_norm_values=True,
    )


def compute_metrics_o1(dataset_folder: Path, result_gep: AnnData) -> Dict[str, float]:
    """
    Load predicted GEP file as pandas DataFrame & compute basic metrics.

    result_gep: Predicted Z' (G x S)

    """

    if not dataset_folder.exists():
        raise FileNotFoundError(f"Datensatzordner nicht gefunden: {dataset_folder}")

    genes = result_gep.obs_names

    # Split genes in marker and non-marker genes
    marker_genes = set(get_shared_genes(dataset_folder))
    result_genes_set = set(genes)
    marker_genes_in_result = list(result_genes_set.intersection(marker_genes))
    non_marker_genes_in_result = list(result_genes_set.difference(marker_genes))

    X = result_gep.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    return compute_basic_metrics_for_gene_groups(
        pd.DataFrame(X, index=result_gep.obs_names, columns=result_gep.var_names),
        marker_genes_in_result,
        non_marker_genes_in_result,
        include_norm_values=True,
    )


def create_norms_histograms(metrics_result: Dict, out_path: Path = None, bins: int = 50, show: bool = False):
    """
    Erzeugt eine einzige Abbildung mit zwei nebeneinander stehenden Histogrammen (PNG):
    links: marker_norms, rechts: non_marker_norms.
    Erwartet, dass metrics_result die Keys 'marker_norms' und 'non_marker_norms' (als Listen) enthält.
    Gibt ein Dict mit dem Pfad der gespeicherten Datei zurück.
    """
    # Normlisten extrahieren
    marker_norms = np.array(metrics_result.get("marker_norms", []), dtype=float)
    non_marker_norms = np.array(metrics_result.get("non_marker_norms", []), dtype=float)

    # Kombinierte Figur mit zwei Subplots nebeneinander
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Marker histogram (links)
    if marker_norms.size > 0:
        axes[0].hist(marker_norms, bins=bins, color="C0", edgecolor="black")
    else:
        axes[0].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[0].set_title("Marker genes - norms histogram")
    axes[0].set_xlabel("Norm")
    axes[0].set_ylabel("Count")

    # Non-marker histogram (rechts)
    if non_marker_norms.size > 0:
        axes[1].hist(non_marker_norms, bins=bins, color="C1", edgecolor="black")
    else:
        axes[1].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[1].set_title("Non-marker genes - norms histogram")
    axes[1].set_xlabel("Norm")
    axes[1].set_ylabel("Count")

    if out_path:
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def create_log_norms_histograms(metrics_result: Dict, out_path: Path = None, bins: int = 50, show: bool = False):
    """
    Erzeugt eine einzige Abbildung mit zwei nebeneinander stehenden Histogrammen (PNG) der Log-Normen:
    links: marker_log_norms, rechts: non_marker_log_norms.
    Erwartet, dass metrics_result die Keys 'marker_norms' und 'non_marker_norms' (als Listen) enthält.
    Gibt ein Dict mit dem Pfad der gespeicherten Datei zurück (oder 'error').
    """
    # Normlisten extrahieren und in numpy arrays
    marker_norms = np.array(metrics_result.get("marker_norms", []), dtype=float)
    non_marker_norms = np.array(metrics_result.get("non_marker_norms", []), dtype=float)

    # Log-Normen erstellen: log(x) für x>0 sonst 0.0 (wie vorher)
    with np.errstate(divide="ignore", invalid="ignore"):
        marker_log = np.where(marker_norms <= 0, 0.0, np.log(marker_norms))
        non_marker_log = np.where(non_marker_norms <= 0, 0.0, np.log(non_marker_norms))

    # Zielordner bestimmen
    # out_path = Path(out_dir)

    # Kombinierte Figur mit zwei Subplots nebeneinander
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Marker log-norm histogram (links)
    if marker_log.size > 0:
        axes[0].hist(marker_log, bins=bins, color="C0", edgecolor="black")
    else:
        axes[0].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[0].set_title("Marker genes - log(norm) histogram")
    axes[0].set_xlabel("log(Norm)")
    axes[0].set_ylabel("Count")

    # Non-marker log-norm histogram (rechts)
    if non_marker_log.size > 0:
        axes[1].hist(non_marker_log, bins=bins, color="C1", edgecolor="black")
    else:
        axes[1].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[1].set_title("Non-marker genes - log(norm) histogram")
    axes[1].set_xlabel("log(Norm)")
    axes[1].set_ylabel("Count")

    if out_path:
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def create_norms_boxplots(metrics_result: Dict, out_path: Path = None, show: bool = False):
    """
    Erzeugt eine Abbildung mit zwei nebeneinander stehenden Boxplots:
    links: marker_norms, rechts: non_marker_norms.
    Speichert als PNG und gibt Dict mit Pfad zurück.
    """
    marker_norms = np.array(metrics_result.get("marker_norms", []), dtype=float)
    non_marker_norms = np.array(metrics_result.get("non_marker_norms", []), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    # Marker boxplot (links)
    if marker_norms.size > 0:
        axes[0].boxplot(marker_norms, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="C0", color="black"),
                        medianprops=dict(color="black"))
        axes[0].set_ylabel("Norm")
    else:
        axes[0].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[0].set_title("Marker genes - norms (boxplot)")
    axes[0].set_xticks([])

    # Non-marker boxplot (rechts)
    if non_marker_norms.size > 0:
        axes[1].boxplot(non_marker_norms, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="C1", color="black"),
                        medianprops=dict(color="black"))
    else:
        axes[1].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[1].set_title("Non-marker genes - norms (boxplot)")
    axes[1].set_xticks([])

    if out_path:
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)


def create_log_norms_boxplots(metrics_result: Dict, out_path: Path = None, show: bool = False):
    """
    Erzeugt eine Abbildung mit zwei nebeneinander stehenden Boxplots der Log-Normen:
    links: marker_log_norms, rechts: non_marker_log_norms.
    Speichert als PNG und gibt Dict mit Pfad zurück.
    """
    marker_norms = np.array(metrics_result.get("marker_norms", []), dtype=float)
    non_marker_norms = np.array(metrics_result.get("non_marker_norms", []), dtype=float)

    # Log-Normen: log(x) für x>0 sonst 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        marker_log = np.where(marker_norms <= 0, 0.0, np.log(marker_norms))
        non_marker_log = np.where(non_marker_norms <= 0, 0.0, np.log(non_marker_norms))

    # Zielordner bestimmen
    # out_path = Path(out_dir)

    saved = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    # Marker log-boxplot (links)
    if marker_log.size > 0:
        axes[0].boxplot(marker_log, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="C0", color="black"),
                        medianprops=dict(color="black"))
        axes[0].set_ylabel("log(Norm)")
    else:
        axes[0].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[0].set_title("Marker genes - log(norm) (boxplot)")
    axes[0].set_xticks([])

    # Non-marker log-boxplot (rechts)
    if non_marker_log.size > 0:
        axes[1].boxplot(non_marker_log, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="C1", color="black"),
                        medianprops=dict(color="black"))
    else:
        axes[1].text(0.5, 0.5, "no data", ha="center", va="center")
    axes[1].set_title("Non-marker genes - log(norm) (boxplot)")
    axes[1].set_xticks([])

    if out_path:
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def main(dataset_folder: Path, result_gep: AnnData, metrics_output_path: Path, compute_scRNA_metrics: bool = False):
    """
    Compute metrics for objective 1 and save results as JSON files / Diagrams.

    - Compute basic metrics on scRNA data + predicted GEP data and save as scRNA_Metrics.json
    - Visualize norms distributions (histograms + boxplots) for scRNA data and predicted GEP data

    Args:
        dataset_folder:
        result_gep: Predicted Z' (G x S)
        metrics_output_path:

    Returns:

    """
    logger.info("Compute metrics for o1")

    if compute_scRNA_metrics:
        # Compute metrics on scRNA data
        res_scrna = compute_metrics_scRNA(dataset_folder)

        # Save result
        metrics_output_path.mkdir(parents=True, exist_ok=True)
        out_path = metrics_output_path / "scRNA_Metrics.json"
        to_save = copy(res_scrna)
        # del to_save["marker_norms"]
        # del to_save["non_marker_norms"]
        with open(out_path, "w") as fh:
            json.dump(
                json.loads(
                    json.dumps(
                        to_save,
                        default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)
                    )
                ),
                fh,
                indent=4,
            )

        # Save result
        scrna_dir = metrics_output_path / "scRNA"
        scrna_dir.mkdir(parents=True, exist_ok=True)

        # Crate some diagrams on metrics of scRNA
        create_norms_histograms(
            res_scrna,
            out_path=scrna_dir / "scRNA_norms_histogram.png",
            show=False,
            bins=100
        )

        create_log_norms_histograms(
            res_scrna,
            out_path=scrna_dir / "scRNA_log_norms_histogram.png",
            show=False,
            bins=100
        )

        create_norms_boxplots(
            res_scrna,
            out_path=scrna_dir / "scRNA_norms_boxplot.png",
            show=False
        )

        create_log_norms_boxplots(
            res_scrna,
            out_path=scrna_dir / "scRNA_log_norms_boxplot.png",
            show=False
        )

    # Compute metrics for o1
    res = compute_metrics_o1(dataset_folder, result_gep)

    # Save result
    metrics_dir = metrics_output_path / "o1"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "o1_metrics.json"
    to_save = copy(res)
    # del to_save["marker_norms"]
    # del to_save["non_marker_norms"]
    with open(out_path, "w") as fh:
        json.dump(
            json.loads(
                json.dumps(
                    to_save,
                    default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)
                )
            ),
            fh,
            indent=4,
        )

    # Crate some diagrams on o1 metrics
    create_norms_histograms(
        res,
        out_path=metrics_dir / "norms_histogram.png",
        show=False,
        bins=100
    )

    create_log_norms_histograms(
        res,
        out_path=metrics_dir / "log_norms_histogram.png",
        show=False,
        bins=100
    )

    create_norms_boxplots(
        res,
        out_path=metrics_dir / "norms_boxplot.png",
        show=False
    )

    create_log_norms_boxplots(
        res,
        out_path=metrics_dir / "log_norms_boxplot.png",
        show=False
    )

