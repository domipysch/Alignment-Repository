import argparse
import sys
import os
import yaml
import torch
import torch.optim as optim
import logging
from anndata import AnnData
from src.utils import load_sc_adata, load_st_adata, fmt_nonzero_4
from src.model import AlternativeIdeaModel
from src.loss import AlternativeIdeaLoss
from src.spatial_graph import build_spatial_graph, SpatialGraphType
from src.dataset import prepare_tensors_from_input
from src.utils import graph_type_from_config
from scipy.sparse import issparse
import pandas as pd
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> tuple[dict, dict, dict, dict]:
    if not os.path.exists(config_path):
        raise Exception(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:

        # Load yaml config
        cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level config must be a mapping (dict).")

        # Ensure required sections exist
        required_sections = ["graph", "model", "training", "loss_weights"]
        missing_sections = [s for s in required_sections if s not in cfg]
        if missing_sections:
            raise ValueError(f"Missing required config sections: {', '.join(missing_sections)}")

        graph_cfg = cfg.get("graph") if isinstance(cfg.get("graph"), dict) else {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        training_cfg = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
        loss_weights_cfg = cfg.get("loss_weights") if isinstance(cfg.get("loss_weights"), dict) else {}

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
        for key in ("lr", "epochs"):
            if key not in training_cfg:
                raise ValueError(f"`training.{key}` is required in the config.")

        # Validate loss_weights is a mapping (specific keys may be optional)
        if not isinstance(loss_weights_cfg, dict):
            raise ValueError("`loss_weights` must be a mapping in the config.")

    return graph_cfg, model_cfg, training_cfg, loss_weights_cfg


def alternative_idea_compute_mapping(
    path_to_config: str,
    adata_sc: AnnData,
    adata_st: AnnData,
    verbose_logging: bool,
    use_device: str = None,
) -> AnnData:

    # 1. Load Config
    logger.debug(f"Load config: {path_to_config}")
    graph_config, model_config, training_config, loss_weights = load_config(path_to_config)
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

    # 3. Convert input anndata to tensors
    logger.debug("Prepare input tensors for model...")
    X, Z, X_shared, Z_shared = prepare_tensors_from_input(adata_sc, adata_st, device)
    logger.info("Prepared input tensors for model.")
    logger.info(f"Shared genes between scRNA and ST: {X_shared.shape[1]}")

    # 4. Build the spatial graph out of Z
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

    # 5. Initialize Model
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
    ).to(device)

    # 6. Initialize Loss and Optimizer
    loss = AlternativeIdeaLoss(
        lambda_rec_spot=loss_weights["lambda_rec_spot"],
        lambda_rec_state=loss_weights["lambda_rec_state"],
        lambda_clust=loss_weights["lambda_clust"],
        lambda_state_entropy=loss_weights["lambda_state_entropy"],
        lambda_spot_entropy=loss_weights["lambda_spot_entropy"],
    )

    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])

    # 7. Training Loop
    logger.info("Starting training loop")
    model.train()

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

        # Logging: verbose -> log every epoch at DEBUG, normal -> log every 10 epochs at INFO
        if verbose_logging:
            logger.debug(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")

    logger.info("Alignment complete.")
    return AnnData(X=A.detach().cpu().numpy())


def main():

    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run AlternativeIdea alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Path to dataset folder')
    parser.add_argument('-c', '--config', dest='config', type=str, help='Path to config.yaml')
    parser.add_argument('-det', '--deterministic_mapping', dest='deterministic_mapping', action="store_true", help='Whether to convert mapping to deterministic (one-hot) form')
    parser.add_argument('--logging', dest='logging', choices=['normal', 'verbose'], default='normal',
                        help="Logging verbosity. Use 'verbose' for more logs.")
    args = parser.parse_args()

    # 2. Configure logging based on argument
    level = logging.DEBUG if args.logging == "verbose" else logging.INFO
    logging.basicConfig(stream=sys.stdout, level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(level)

    # Step 3: Load data
    logger.debug("Load input scRNA an ST data...")
    adata_sc = load_sc_adata(args.dataset)  # C x G
    adata_st = load_st_adata(args.dataset)  # S x G
    logger.info("Loaded input scRNA and ST data.")

    # Step 4: Map data using AlternativeIdea
    spot_to_cell_map = alternative_idea_compute_mapping(
        args.config,
        adata_sc,
        adata_st,
        verbose_logging=(args.logging == "verbose"),
        use_device="cpu"
    )  # S x C
    logger.info("Obtained spot-to-cell mapping AnnData.")
    assert spot_to_cell_map.X.shape == (adata_st.n_obs, adata_sc.n_obs), "dims passen nicht"

    # Step 5 (optional): Apply one-hot encoding to mapping
    logger.info("Apply deterministic mapping" if args.deterministic_mapping else "Keep probabilistic mapping")
    if args.deterministic_mapping:

        # For each row (spot) in ad_map, set the max value to 1 and all others to 0
        if issparse(spot_to_cell_map.X):
            mat = torch.from_numpy(spot_to_cell_map.X.toarray())
        else:
            mat = torch.as_tensor(spot_to_cell_map.X)

        argmax_idx = torch.argmax(mat, dim=1, keepdim=True)  # for each spot (column), index of max cell / cell type
        one_hot = torch.zeros_like(mat)
        one_hot.scatter_(1, argmax_idx, 1.0)
        spot_to_cell_map.X = one_hot.detach().cpu().numpy()

    # Step 6: Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
    predicted_spot_expressions = spot_to_cell_map.X @ adata_sc.X  # S x G

    # Transpose to G x S
    predicted_spot_expressions = predicted_spot_expressions.transpose()  # now G x S
    assert predicted_spot_expressions.shape == (adata_sc.n_vars, adata_st.n_obs), "dims passen nicht"

    # Step 6: Export predicted_spot_expressions to CSV
    # - Rows: Genes
    # - Columns: Spots
    # - Top left cell = "GEP"
    output_path = os.path.join(args.dataset, "results_alternative_idea", "alternative_idea_gep.csv")
    logger.info(f"Write result GEP to CSV: {output_path}")
    df = pd.DataFrame(
        predicted_spot_expressions,
        index=list(s.upper() for s in adata_sc.var_names),
        columns=adata_st.obs_names,
    )
    df_formatted = df.map(fmt_nonzero_4)
    df_formatted.to_csv(output_path, index=True, index_label="GEP")  # "GEP" in cell 0,0
    logger.info(f"Saved tangram GEP to {output_path}")


if __name__ == "__main__":
    main()

