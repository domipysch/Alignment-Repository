import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_summary(results_dir: Path) -> dict:
    """
    Load summary.csv from a results directory.

    Args:
        results_dir: Path to the experiment results folder containing summary.csv.

    Returns:
        Dict mapping run_id (str) to a dict with keys 'L1', 'L4', 'L5' (floats).
    """
    summary_path = results_dir / "summary.csv"
    df = pd.read_csv(summary_path)
    result = {}
    for _, row in df.iterrows():
        run_id = str(int(row["id"]))
        result[run_id] = {
            "L1": float(row["L1"]),
            "L4": float(row["L4"]),
            "L5": float(row["L5"]),
        }
    return result


def load_hyperparams(results_dir: Path, run_id: str) -> tuple[float, float]:
    """
    Load the entropy loss weights from config.yml for a single run.

    Args:
        results_dir: Path to the experiment results folder.
        run_id: Subdirectory name of the run (e.g. '0', '1', ...).

    Returns:
        Tuple (lambda_state_entropy, lambda_spot_entropy) as floats.
    """
    config_path = results_dir / run_id / "config.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    lw = config["loss_weights"]
    return float(lw["lambda_state_entropy"]), float(lw["lambda_spot_entropy"])


def build_grid(results_dir: Path, summary: dict) -> tuple:
    """
    Build sorted axis values and 2D loss matrices for heatmap plotting.

    Args:
        results_dir: Path to the experiment results folder (used to read per-run configs).
        summary: Dict mapping run_id to loss dict, as returned by load_summary.

    Returns:
        Tuple (state_axis, spot_axis, L1_grid, L4_grid, L5_grid), where the grids
        are 2D arrays of shape (n_state_vals, n_spot_vals).
    """
    state_vals = set()
    spot_vals = set()
    run_params = {}

    for run_id in summary:
        state_entropy, spot_entropy = load_hyperparams(results_dir, run_id)
        state_vals.add(state_entropy)
        spot_vals.add(spot_entropy)
        run_params[run_id] = (state_entropy, spot_entropy)

    state_axis = sorted(state_vals)
    spot_axis = sorted(spot_vals)
    n_state = len(state_axis)
    n_spot = len(spot_axis)

    state_idx = {v: i for i, v in enumerate(state_axis)}
    spot_idx = {v: i for i, v in enumerate(spot_axis)}

    L1_grid = np.full((n_state, n_spot), np.nan)
    L4_grid = np.full((n_state, n_spot), np.nan)
    L5_grid = np.full((n_state, n_spot), np.nan)

    for run_id, losses in summary.items():
        state_entropy, spot_entropy = run_params[run_id]
        i = state_idx[state_entropy]
        j = spot_idx[spot_entropy]
        L1_grid[i, j] = losses["L1"]
        L4_grid[i, j] = losses["L4"]
        L5_grid[i, j] = losses["L5"]

    return state_axis, spot_axis, L1_grid, L4_grid, L5_grid


def plot_heatmap(ax, matrix, state_axis, spot_axis, title):
    """
    Plot a single annotated heatmap panel with a colorbar.

    Args:
        ax: Matplotlib Axes to draw on.
        matrix: 2D array of shape (n_spot_vals, n_state_vals) to visualize.
        state_axis: Sorted list of lambda_state_entropy values (x-axis ticks).
        spot_axis: Sorted list of lambda_spot_entropy values (y-axis ticks).
        title: Title string for the subplot.
    """
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(state_axis)))
    ax.set_xticklabels([f"{v:.3g}" for v in state_axis])
    ax.set_yticks(range(len(spot_axis)))
    ax.set_yticklabels([f"{v:.3g}" for v in spot_axis])

    ax.set_xlabel("λ_state_entropy", fontsize=13)
    ax.set_ylabel("λ_spot_entropy", fontsize=13)
    ax.set_title(title, fontsize=14)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white",
                )


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss heatmaps for grid search over L4/L5 weights."
    )
    parser.add_argument(
        "-r", "--results", required=True, help="Path to experiment results folder"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save the output PNG"
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_path = Path(args.output)

    summary = load_summary(results_dir)
    state_axis, spot_axis, L1_grid, L4_grid, L5_grid = build_grid(results_dir, summary)

    # Transpose so rows=spot_entropy (y-axis), cols=state_entropy (x-axis)
    L1_plot = L1_grid.T
    L4_plot = L4_grid.T
    L5_plot = L5_grid.T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_heatmap(axes[0], L1_plot, state_axis, spot_axis, "L_rec_spot")
    plot_heatmap(axes[1], L4_plot, state_axis, spot_axis, "L_state_entropy")
    plot_heatmap(axes[2], L5_plot, state_axis, spot_axis, "L_spot_entropy")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved heatmap to {output_path}")

    plt.show()


if __name__ == "__main__":
    """
    Visualize state_entropy and spot_entropy loss weights effect
    on final loss values across a grid search experiment.

    Usage:
        python -m MPA_Code.metrics.utils.create_loss_heatmap \
            -r <results_folder> \
            -o <output.pdf>
    """
    main()
