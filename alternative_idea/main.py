import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List
import yaml
import scanpy as sc
import torch
import torch.optim as optim
import logging
from anndata import AnnData
from MPA_Code.utils.io import anndata_to_csv
from MPA_Code.alternative_idea.src.utils import load_sc_adata, load_st_adata, fmt_nonzero_4
from MPA_Code.alternative_idea.src.model import AlternativeIdeaModel
from MPA_Code.alternative_idea.src.loss import AlternativeIdeaLoss
from MPA_Code.alternative_idea.src.spatial_graph import build_spatial_graph, SpatialGraphType
from MPA_Code.alternative_idea.src.dataset import prepare_tensors_from_input
from MPA_Code.alternative_idea.src.utils import graph_type_from_config
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    if not os.path.exists(config_path):
        raise Exception(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:

        # Load yaml config
        cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level config must be a mapping (dict).")

        # Ensure required sections exist
        required_sections = ["mapping", "graph", "model", "training", "loss_weights"]
        missing_sections = [s for s in required_sections if s not in cfg]
        if missing_sections:
            raise ValueError(f"Missing required config sections: {', '.join(missing_sections)}")

        mapping_cfg = cfg.get("mapping") if isinstance(cfg.get("mapping"), dict) else {}
        graph_cfg = cfg.get("graph") if isinstance(cfg.get("graph"), dict) else {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
        loss_weights_cfg = cfg.get("loss_weights") if isinstance(cfg.get("loss_weights"), dict) else {}

        # Validate mapping config
        if not mapping_cfg:
            raise ValueError("`mapping` must be a mapping in the config.")
        for key in ("deterministic",):
            if key not in mapping_cfg:
                raise ValueError(f"`mapping.{key}` is required in the config.")

        # Validate graph config
        if not graph_cfg:
            raise ValueError("`graph` must be a mapping in the config.")
        graph_type = graph_cfg.get("type")
        if not isinstance(graph_type, str):
            raise ValueError("`graph.type` must be specified as a string (e.g. 'knn', 'mutual_knn', 'radius').")
        graph_type_l = graph_type.lower()
        if graph_type_l in ("knn", "mutual_knn"):
            if "k" not in graph_cfg:
                raise ValueError(f"`graph.k` must be specified for graph.type='{graph_type}'.")
            if not isinstance(graph_cfg["k"], int) or graph_cfg["k"] <= 0:
                raise ValueError("`graph.k` must be a positive integer.")
        elif graph_type_l == "radius":
            if "radius" not in graph_cfg:
                raise ValueError("`graph.radius` must be specified for graph.type='radius'.")
            try:
                radius_val = float(graph_cfg["radius"])
                if radius_val <= 0:
                    raise ValueError("`graph.radius` must be a positive number.")
            except Exception:
                raise ValueError("`graph.radius` must be a numeric value.")

        # Validate model config
        if not model_cfg:
            raise ValueError("`model` must be a mapping in the config.")
        for key in ("d", "K", "enc_hidden_dim", "dec_hidden_dim"):
            if key not in model_cfg:
                raise ValueError(
                    f"`model.{key}` is required in the config.")

        # Validate training config
        if not training_cfg:
            raise ValueError("`training` must be a mapping in the config.")
        for key in ("lr", "epochs", "dropout_decoder"):
            if key not in training_cfg:
                raise ValueError(f"`training.{key}` is required in the config.")

        # Validate loss_weights is a mapping (specific keys may be optional)
        if not isinstance(loss_weights_cfg, dict):
            raise ValueError("`loss_weights` must be a mapping in the config.")

    return mapping_cfg, graph_cfg, model_cfg, training_cfg, loss_weights_cfg


def alternative_idea_compute_mapping(
    path_to_config: Path,
    adata_sc: AnnData,
    adata_st: AnnData,
    verbose_logging: bool,
    use_device: str = None,
) -> tuple[torch.Tensor, List[float]]:

    # 1. Load Config
    logger.debug(f"Load config: {path_to_config}")
    mapping_config, graph_config, model_config, training_config, loss_weights = load_config(path_to_config)
    logger.info(f"Loaded config: {path_to_config}")
    logger.debug(f"Loaded training config: {training_config}")
    logger.debug(f"Loaded model config: {model_config}")
    logger.debug(f"Loaded graph config: {graph_config}")
    logger.debug(f"Loaded loss weights: {loss_weights}")

    # 2. Setup Device
    # Use 'mps' for Apple Silicon, 'cuda' for NVIDIA, or 'cpu'
    if use_device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(use_device)
    logger.info(f"Using device: {device}")

    # 3. Preprocess data (only for mapping): Normalize & Log-transform
    logger.info("Normalize & Log-transform gene expression and spatial data")
    sc.pp.normalize_total(adata_sc)
    sc.pp.normalize_total(adata_st)
    sc.pp.log1p(adata_sc)
    sc.pp.log1p(adata_st)

    # 4. Convert input anndata to tensors
    logger.debug("Prepare input tensors for model...")
    X, Z, X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)
    logger.info("Prepared input tensors for model.")
    logger.info(f"Shared genes between scRNA and ST: {X_shared.shape[1]}")

    # 5. Build the spatial graph out of Z
    graph_type = graph_type_from_config(graph_config)
    k = graph_config["k"] if graph_type in (SpatialGraphType.KNN, SpatialGraphType.MUTUAL_KNN) else None
    radius = graph_config["radius"] if graph_type == SpatialGraphType.RADIUS else None
    edge_index = build_spatial_graph(
        adata_st,
        method=graph_type,
        k=k,
        radius=radius,
        device=device,
    )
    logger.info(f"Created spatial graph of type {graph_config['type']}.")

    # 6. Initialize Model
    # Dimensions derived from tensors
    num_spots, g_st = Z.shape
    num_cells, g_sc = X.shape
    num_genes_shared = X_shared.shape[1]
    logger.debug(f"ST dimensions: num_spots={num_spots}, g_st={g_st}")
    logger.debug(f"scRNA dimensions: num_cells={num_cells}, g_sc={g_sc}")
    logger.debug(f"Number of genes shared: {num_genes_shared}")

    model = AlternativeIdeaModel(
        num_spots_st=num_spots,
        num_cells_sc=num_cells,
        g_st=g_st,
        g_sc=g_sc,
        d=model_config["d"],
        k=model_config["K"],
        enc_hidden_dim=model_config["enc_hidden_dim"],
        dec_hidden_dim=model_config["dec_hidden_dim"],
        dropout_rate_decoder=training_config["dropout_decoder"],
    ).to(device)

    # 7. Initialize Loss and Optimizer
    loss = AlternativeIdeaLoss(
        lambda_rec_spot=loss_weights["lambda_rec_spot"],
        lambda_rec_state=loss_weights["lambda_rec_state"],
        lambda_clust=loss_weights["lambda_clust"],
        lambda_state_entropy=loss_weights["lambda_state_entropy"],
        lambda_spot_entropy=loss_weights["lambda_spot_entropy"],
    )

    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])

    # 8. Training Loop
    logger.info("Starting optimization loop")
    model.train()

    losses: List[float] = []
    for epoch in range(int(training_config["epochs"])):
        optimizer.zero_grad()

        # Forward pass
        A, B, h, M_rec, F = model(Z, edge_index)

        # Calculate segmented losses
        loss_dict = loss(
            A=A,
            B=B,
            h=h,
            M_rec=M_rec,
            F=F,
            X=X,
            X_shared=X_shared,
            Z_shared=Z_shared
        )

        total_loss = loss_dict["loss"]

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Collect loss value
        losses.append(float(total_loss.item()))

        # Logging: verbose -> log every epoch at DEBUG, normal -> log every 10 epochs at INFO
        if verbose_logging:
            logger.debug(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")

    logger.info("Alignment complete.")
    return A, losses


def main(
    dataset_folder: Path,
    config_path: Path,
    output_path: Optional[Path],
    verbose_logging: bool = False
) -> AnnData:

    TORCH_DEVICE = "mps"

    # Step 1: Load data
    logger.info("Load input scRNA and ST data...")
    adata_sc = load_sc_adata(dataset_folder)  # C x G
    adata_st = load_st_adata(dataset_folder)  # S x G
    logger.info("Loaded input scRNA and ST data.")

    # Step 2: Map data using AlternativeIdea
    spot_to_cell_map: torch.Tensor
    spot_to_cell_map, losses = alternative_idea_compute_mapping(
        config_path,
        adata_sc.copy(),
        adata_st.copy(),
        verbose_logging=verbose_logging,
        use_device=TORCH_DEVICE
    )  # S x C, plus loss history
    logger.info("Obtained spot-to-cell mapping AnnData.")
    assert spot_to_cell_map.shape == (adata_st.n_obs, adata_sc.n_obs), "dims passen nicht"

    # Create and save loss curve figure
    loss_fig_path = config_path.parent / "loss–curve.png"
    plt.figure()
    plt.plot(range(len(losses)), losses, marker='o', linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(loss_fig_path))
    plt.close()
    logger.info(f"Saved loss curve to {loss_fig_path}")

    # Step 3 (optional): Apply one-hot encoding to mapping
    mapping_config, _, _, _, _ = load_config(config_path)
    logger.info("Apply deterministic mapping" if mapping_config["deterministic"] else "Keep probabilistic mapping")
    if mapping_config["deterministic"]:
        argmax_idx = torch.argmax(spot_to_cell_map, dim=1, keepdim=True)  # for each spot (column), index of max cell / cell type
        spot_to_cell_map = torch.zeros_like(spot_to_cell_map)
        spot_to_cell_map.scatter_(1, argmax_idx, 1.0)

    # Step 4: Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
    adata_sc_tensor = torch.as_tensor(adata_sc.X, dtype=torch.float32, device=TORCH_DEVICE)
    predicted_spot_expressions = spot_to_cell_map @ adata_sc_tensor  # S x G

    # Transpose to G x S
    predicted_spot_expressions = predicted_spot_expressions.T  # now G x S
    assert predicted_spot_expressions.shape == (adata_sc.n_vars, adata_st.n_obs), "dims passen nicht"

    # Create AnnData object for predicted spot expressions
    adata_result = AnnData(X=predicted_spot_expressions.detach().cpu().numpy())
    adata_result.obs_names = adata_sc.var_names
    adata_result.var_names = adata_st.obs_names

    # Step 5 (optional): Export predicted_spot_expressions to CSV
    # - Rows: Genes
    # - Columns: Spots
    if output_path is not None:
        logger.info(f"Write result GEP to CSV: {output_path}")
        anndata_to_csv(
            adata_result,
            output_path,
            top_left_label="GEP",
            format_func=fmt_nonzero_4,
            uppercase_var_names=True,
        )
        logger.info(f"Saved result GEP to {output_path}")
    else:
        logger.debug("No output path provided, skipping CSV export.")

    # Step 6: Return result
    return adata_result


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Run AlternativeIdea alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', dest='dataset', type=Path, help='Path to dataset folder')
    parser.add_argument('-c', '--config', dest='config', type=Path, help='Path to config.yaml')
    parser.add_argument('-o', '--output_path', dest='output_path', type=Path, required=False, default=None, help='Path where to store result to')
    parser.add_argument('--logging', dest='logging', choices=['normal', 'verbose'], default='normal',
                        help="Logging verbosity. Use 'verbose' for more logs.")
    args = parser.parse_args()

    # Configure logging based on argument
    level = logging.DEBUG if args.logging == "verbose" else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(level)

    # Run alignment
    main(
        args.dataset,
        args.config,
        args.output_path,
        verbose_logging=(args.logging == "verbose"),
    )

