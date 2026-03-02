from pathlib import Path
from typing import Dict
from .spatial_graph import SpatialGraphType
import anndata as ad
import pandas as pd
import logging
from Code.utils.io import csv_to_anndata
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_sc_adata(dataset_folder: Path) -> ad.AnnData:
    """
    Load single-cell data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
    Returns:
        ad.AnnData: Single-cell AnnData object (C x G)
    """
    logger.debug("Load scRNA data")
    # In file: G x C, we want: C x G
    return csv_to_anndata(dataset_folder / "scData_GEP.csv", transpose=True)


def load_st_adata(dataset_folder: Path) -> ad.AnnData:
    """
    Load ST data from dataset folder into an AnnData object.
    Args:
        dataset_folder: Absolute path to dataset folder
    Returns:
        ad.AnnData: ST AnnData object (S x G)
    """
    logger.debug("Load ST data")
    # In file: G x S, we want: S x G
    adata_st = csv_to_anndata(dataset_folder / "stData_GEP.csv", transpose=True)
    # Load spot coordinates
    logger.debug("Load ST coordinates")
    coords = pd.read_csv(dataset_folder / "stData_Spots.csv", index_col=0)
    adata_st.obsm["spatial"] = coords[["cArray0", "cArray1"]].values
    return adata_st


def fmt_nonzero_4(x: float) -> str:
    """
    Format a numeric value for display to cap at up to four decimal places.

    Args:
        x: Input value (float)
    Returns:
        str: Formatted string
    """
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
    except Exception:
        raise Exception("Input value is not convertible to float")
    if xf == 0.0:
        return "0.0"
    return f"{xf:.4f}"


def graph_type_from_config(graph_cfg: Dict) -> SpatialGraphType:

    if not isinstance(graph_cfg, dict):
        raise ValueError("`graph_cfg` must be a dict.")

    graph_type = graph_cfg.get("type")
    if not isinstance(graph_type, str):
        raise ValueError("`graph.type` must be a string in the config.")

    t = graph_type.strip().lower()

    if t == "knn":
        return SpatialGraphType.KNN
    if t == "mutual_knn":
        return SpatialGraphType.MUTUAL_KNN
    if t == "radius":
        return SpatialGraphType.RADIUS
    if t == "delaunay":
        return SpatialGraphType.DELAUNAY

    raise ValueError(
        f"Unsupported graph.type: '{graph_type}'. Expected one of: knn, mutual_knn, radius, delaunay."
    )


def dump_loss_logs(losses, config_path) -> dict:

    losses_after_last_epoch = {}
    for comp in (
        "rec_spot",
        "rec_gene",
        "rec_state",
        "clust",
        "state_entropy",
        "spot_entropy",
    ):
        comp_vals = losses.get(comp, {})
        val = None
        if isinstance(comp_vals, dict):
            vals_list = comp_vals.get("values", [])
            weight = comp_vals.get("weight")
            if len(vals_list) > 0:
                val = vals_list[-1]

        # Round unweighted final value to 2 decimals for clarity (handle None)
        losses_after_last_epoch[f"{comp}"] = (
            round(float(val), 2) if val is not None else None
        )

    loss_dir = config_path.parent / "loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    df_end = pd.DataFrame([losses_after_last_epoch])
    df_end.to_csv(loss_dir / "losses_end.csv", index=False)
    logger.info(f"Saved final loss values to {loss_dir / 'losses_end.csv'}")

    return losses_after_last_epoch


def create_loss_plots(losses, loss_dir):

    loss_dir.mkdir(parents=True, exist_ok=True)

    loss_fig_path = loss_dir / "loss–curves-weighted.png"
    plt.figure()
    # Plot individual components + total
    epochs = list(range(len(losses["total-weighted"])))
    plt.plot(
        epochs,
        losses["total-weighted"],
        label="total-weighted",
        linewidth=2,
        color="black",
    )
    plt.plot(
        epochs,
        list(v * losses["rec_spot"]["weight"] for v in losses["rec_spot"]["values"]),
        label="rec_spot-weighted",
    )
    plt.plot(
        epochs,
        list(v * losses["rec_gene"]["weight"] for v in losses["rec_gene"]["values"]),
        label="rec_gene-weighted",
    )
    plt.plot(
        epochs,
        list(v * losses["rec_state"]["weight"] for v in losses["rec_state"]["values"]),
        label="rec_state-weighted",
    )
    plt.plot(
        epochs,
        list(v * losses["clust"]["weight"] for v in losses["clust"]["values"]),
        label="clust-weighted",
    )
    plt.plot(
        epochs,
        list(
            v * losses["state_entropy"]["weight"]
            for v in losses["state_entropy"]["values"]
        ),
        label="state_entropy-weighted",
    )
    plt.plot(
        epochs,
        list(
            v * losses["spot_entropy"]["weight"]
            for v in losses["spot_entropy"]["values"]
        ),
        label="spot_entropy-weighted",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (components + total)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(loss_fig_path))
    plt.close()
    logger.info(f"Saved loss curve to {loss_fig_path}")

    num_epochs = len(losses.get("total-weighted", []))

    # Components (order for plotting)
    components = (
        "rec_spot",
        "rec_gene",
        "rec_state",
        "clust",
        "state_entropy",
        "spot_entropy",
    )

    # Ensure epochs list is available
    epochs = list(range(num_epochs))

    # Save one plot per loss component
    for comp in components:
        plt.figure()
        y = losses[comp]["values"]
        plt.plot(epochs, y, label=comp, linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss: {comp} - unweighted")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = loss_dir / f"{comp}.png"
        plt.savefig(str(out_path))
        plt.close()
        logger.info(f"Saved per-loss plot to {out_path}")
