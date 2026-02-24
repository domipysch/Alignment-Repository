import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import scanpy as sc
import torch
import torch.optim as optim
import logging
from anndata import AnnData
from MPA_Code.utils.io import anndata_to_csv
from MPA_Code.alternative_idea.src.utils import load_sc_adata, load_st_adata, fmt_nonzero_4, create_loss_plots, \
    dump_loss_logs
from MPA_Code.alternative_idea.src.model import AlternativeIdeaModel
from MPA_Code.alternative_idea.src.loss import AlternativeIdeaLoss
from MPA_Code.alternative_idea.src.spatial_graph import build_spatial_graph, SpatialGraphType
from MPA_Code.alternative_idea.src.dataset import prepare_tensors_from_input
from MPA_Code.alternative_idea.src.utils import graph_type_from_config
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
        for key in ("lr", "epochs", "dropout_decoder", "use_cm"):
            if key not in training_cfg:
                raise ValueError(f"`training.{key}` is required in the config.")
        # Optional normalize_and_log flag: default True (preserve previous behavior)
        if "normalize_and_log" in training_cfg:
            if not isinstance(training_cfg["normalize_and_log"], bool):
                raise ValueError("`training.normalize_and_log` must be a boolean if provided.")
        else:
            training_cfg["normalize_and_log"] = True

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
    save_intermediate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:

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

    # (Optional) 3. Preprocess data (only for mapping): Normalize & Log-transform
    if training_config["normalize_and_log"]:
        logger.info("Normalize & Log-transform gene expression and spatial data")
        sc.pp.normalize_total(adata_sc)
        sc.pp.normalize_total(adata_st)
        sc.pp.log1p(adata_sc)
        sc.pp.log1p(adata_st)
    else:
        logger.info("Skipping normalize_and_log as per config (training.normalize_and_log=false)")

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
        lambda_rec_gene=loss_weights["lambda_rec_gene"],
        lambda_rec_state=loss_weights["lambda_rec_state"],
        lambda_clust=loss_weights["lambda_clust"],
        lambda_state_entropy=loss_weights["lambda_state_entropy"],
        lambda_spot_entropy=loss_weights["lambda_spot_entropy"],
        normalize_by_initial=True,
        warmup_iters=1,
        k=model_config["K"],
        use_cm=bool(training_config["use_cm"]),
    )

    optimizer = optim.Adam(model.parameters(), lr=training_config["lr"])

    # 8. Training Loop
    logger.info("Starting optimization loop")
    model.train()

    # Collect per-epoch losses for plotting: individual components + total
    losses = {
        "total-weighted": [],
        "rec_spot": {
            "weight": loss_weights["lambda_rec_spot"],
            "values": []
        },
        "rec_gene": {
            "weight": loss_weights["lambda_rec_gene"],
            "values": []
        },
        "rec_state": {
            "weight": loss_weights["lambda_rec_state"],
            "values": []
        },
        "clust": {
            "weight": loss_weights["lambda_clust"],
            "values": []
        },
        "state_entropy": {
            "weight": loss_weights["lambda_state_entropy"],
            "values": []
        },
        "spot_entropy": {
            "weight": loss_weights["lambda_spot_entropy"],
            "values": []
        }
    }

    def to_scalar(t):
        try:
            if torch.is_tensor(t):
                return float(t.detach().cpu().item())
            else:
                return float(t)
        except Exception:
            return float(t)

    # Initialize A so we can safely return it even when epochs == 0
    A = None
    B = None

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

        # 3. Optionaler Gradient-Check (Nur zur Diagnose)
        if epoch % 100 == 0:
            logger.debug(f"\n--- Gradient Analysis (Epoch {epoch}) ---")
            for name, loss_val in loss_dict.items():
                if name == "loss": continue

                # Temporäre Gradientenberechnung für diesen spezifischen Term
                model.zero_grad()
                loss_val.backward(retain_graph=True)

                # Berechne die Norm über alle trainierbaren Gewichte
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
                logger.debug(f"Term: {name:15} | Loss: {loss_val.item():.4f} | Grad-Norm: {grad_norm:.4f}")

            # Reset für den eigentlichen Backprop
            model.zero_grad()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Collect loss values
        losses["total-weighted"].append(to_scalar(total_loss))
        losses["rec_spot"]["values"].append(to_scalar(loss_dict.get("rec_spot")))
        losses["rec_gene"]["values"].append(to_scalar(loss_dict.get("rec_gene")))
        losses["rec_state"]["values"].append(to_scalar(loss_dict.get("rec_state")))
        losses["clust"]["values"].append(to_scalar(loss_dict.get("clust")))
        losses["state_entropy"]["values"].append(to_scalar(loss_dict.get("state_entropy")))
        losses["spot_entropy"]["values"].append(to_scalar(loss_dict.get("spot_entropy")))

        # Logging: verbose -> log every epoch at DEBUG, normal -> log every 10 epochs at INFO
        if verbose_logging:
            logger.debug(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")
        else:
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f}")

    if save_intermediate:
        logger.info("Saving intermediate results...")
        folder_intermediate = path_to_config.parent / "intermediate"
        folder_intermediate.mkdir(parents=True, exist_ok=True)

        # df = pd.DataFrame(X.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "X.csv", index=False)

        # df = pd.DataFrame(Z.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "Z.csv", index=False)

        # df = pd.DataFrame(X_shared.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "X_shared.csv", index=False)

        # df = pd.DataFrame(Z_shared.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "Z_shared.csv", index=False)

        # df = pd.DataFrame(A.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "A.csv", index=False)

        # df = pd.DataFrame(B.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "B.csv", index=False)

        C = torch.matmul(A, B)
        # df = pd.DataFrame(C.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "C.csv", index=False)

        A_thresh = A.detach().cpu().clone()
        A_thresh[A_thresh < 0.1] = 0.0
        df_weight = pd.DataFrame(A_thresh.numpy())
        df_weight.to_csv(folder_intermediate / 'A_thresh.csv', index=False)

        B_thresh = B.detach().cpu().clone()
        B_thresh[B_thresh < 0.1] = 0.0
        df_weight = pd.DataFrame(B_thresh.numpy())
        df_weight.to_csv(folder_intermediate / 'B_thresh.csv', index=False)

        C_thresh = C.detach().cpu().clone()
        C_thresh[C_thresh < 0.1] = 0.0
        df_weight = pd.DataFrame(C_thresh.numpy())
        df_weight.to_csv(folder_intermediate / 'C_thresh.csv', index=False)


        # idx = torch.argmax(A, dim=1, keepdim=True)
        # A_argmax = torch.zeros_like(A).scatter_(1, idx, 1.0)
        # pd.DataFrame(A_argmax.detach().cpu().numpy()).to_csv(folder_intermediate / "A_argmax.csv", index=False)

        # idx = torch.argmax(B, dim=1, keepdim=True)
        # B_argmax = torch.zeros_like(B).scatter_(1, idx, 1.0)
        # pd.DataFrame(B_argmax.detach().cpu().numpy()).to_csv(folder_intermediate / "B_argmax.csv", index=False)

        # idx = torch.argmax(C, dim=1, keepdim=True)
        # C_argmax = torch.zeros_like(C).scatter_(1, idx, 1.0)
        # pd.DataFrame(C_argmax.detach().cpu().numpy()).to_csv(folder_intermediate / "C_argmax.csv", index=False)

        p = torch.mean(B, dim=0)
        pd.DataFrame(p.detach().cpu().numpy()).to_csv(folder_intermediate / "p.csv", index=False)

        # B_normalized = B / (torch.sum(B, dim=0) + 1e-6)  # (K x C)
        # df_weight = pd.DataFrame(B_normalized.detach().cpu().numpy())
        # df_weight.to_csv(folder_intermediate / 'B_normalized.csv', index=False)

        # M = torch.matmul(B.t(), X)
        # df = pd.DataFrame(M.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "M.csv", index=False)

        # M_normalized = torch.matmul(B_normalized.t(), X)
        # df = pd.DataFrame(M_normalized.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "M_normalized.csv", index=False)

        # Z_prime = torch.matmul(A, X)
        # df = pd.DataFrame(Z_prime.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "Z_prime.csv", index=False)

        # CM_normalized = torch.matmul(C, M_normalized)
        # df = pd.DataFrame(CM_normalized.detach().cpu().numpy())
        # df.to_csv(folder_intermediate / "CM_normalized.csv", index=False)

        if mapping_config.get("deterministic"):

            # B: Argmax per cell -> One-hot pro Zeile
            argmax_idx = torch.argmax(B, dim=1, keepdim=True)
            B_argmax = torch.zeros_like(B)
            B_argmax.scatter_(1, argmax_idx, 1.0)
            df_weight = pd.DataFrame(B_argmax.detach().cpu().numpy())
            df_weight.to_csv(folder_intermediate / 'B_argmax.csv', index=False)

            # C: Argmax per spot -> One-hot pro Zeile
            C_argmax = torch.matmul(A, B_argmax)
            argmax_idx = torch.argmax(C_argmax, dim=1, keepdim=True)
            C_argmax = torch.zeros_like(C_argmax)
            C_argmax.scatter_(1, argmax_idx, 1.0)
            df_weight = pd.DataFrame(C_argmax.detach().cpu().numpy())
            df_weight.to_csv(folder_intermediate / 'C_argmax.csv', index=False)

            # todo. ggf. auch andere det dumpen.

    logger.info("Alignment complete.")
    return A, B, losses


def compute_gene_expression_prediction(
    spot_to_cell_map: torch.Tensor,
    cell_to_cell_type: torch.Tensor,
    adata_sc: AnnData,
    adata_st: AnnData,
    deterministic_mapping: bool,
    torch_device: str,
    use_cm: bool,
) -> AnnData:

    adata_sc_tensor = torch.as_tensor(adata_sc.X, dtype=torch.float32, device=torch_device)

    if deterministic_mapping:

        # Leave A is it is
        pass

        # B: Argmax per cell -> One-hot pro Zeile
        argmax_idx = torch.argmax(cell_to_cell_type, dim=1, keepdim=True)
        cell_to_cell_type = torch.zeros_like(cell_to_cell_type)
        cell_to_cell_type.scatter_(1, argmax_idx, 1.0)

        # C
        C = torch.matmul(spot_to_cell_map, cell_to_cell_type)  # S x T
        argmax_idx = torch.argmax(C, dim=1, keepdim=True)
        C_argmax = torch.zeros_like(C)
        C_argmax.scatter_(1, argmax_idx, 1.0)

    else:
        # A, B leave is they are
        C = torch.matmul(spot_to_cell_map, cell_to_cell_type)  # S x T

    if use_cm:
        logger.info("Using CM for final mapping output as per config (training.use_cm=true)")
        # Compute M = B_normalized^T @ X
        B_normalized = cell_to_cell_type / (torch.sum(cell_to_cell_type, dim=0) + 1e-6)  # (K x C)
        M = torch.matmul(B_normalized.t(), adata_sc_tensor)  # T x G
        # Compute Z' = C @ M
        predicted_spot_expressions = torch.matmul(C, M)  # S x G
    else:
        predicted_spot_expressions = torch.matmul(spot_to_cell_map, adata_sc_tensor)  # S x G

    # Transpose to G x S
    predicted_spot_expressions = predicted_spot_expressions.T  # now G x S
    assert predicted_spot_expressions.shape == (adata_sc.n_vars, adata_st.n_obs), "dims passen nicht"

    # Create AnnData object for predicted spot expressions
    adata_result = AnnData(X=predicted_spot_expressions.detach().cpu().numpy())
    adata_result.obs_names = adata_sc.var_names
    adata_result.var_names = adata_st.obs_names

    return adata_result


def main(
    dataset_folder: Path,
    config_path: Path,
    output_path: Optional[Path],
    mapping_output_path: Optional[Path] = None,
    verbose_logging: bool = False
) -> tuple[AnnData, Optional[AnnData], dict]:

    mapping_config, _, _, training_config, _ = load_config(config_path)

    TORCH_DEVICE = "mps"

    # Step 1: Load data
    logger.info("Load input scRNA and ST data...")
    adata_sc = load_sc_adata(dataset_folder)  # C x G
    adata_st = load_st_adata(dataset_folder)  # S x G
    logger.info("Loaded input scRNA and ST data.")

    # Step 2: Map data using AlternativeIdea
    spot_to_cell_map: torch.Tensor
    spot_to_cell_map, cell_to_cell_type, losses = alternative_idea_compute_mapping(
        config_path,
        adata_sc.copy(),
        adata_st.copy(),
        verbose_logging=verbose_logging,
        use_device=TORCH_DEVICE,
        save_intermediate=True,  # todo. temp.
    )  # S x C, plus loss history
    logger.info("Obtained spot-to-cell mapping AnnData.")
    assert spot_to_cell_map.shape == (adata_st.n_obs, adata_sc.n_obs), "dims passen nicht"

    # Step 3.1: Save end values of loss terms (value after last epoch) to a csv file
    losses_after_last_epoch = dump_loss_logs(losses, config_path)

    # Step 3.2: Create plots for loss curves
    create_loss_plots(losses, config_path.parent / "loss")

    # Step 3.3: Save mapping to CSV
    if mapping_output_path is not None:
        logger.info(f"Write spot-to-cell mapping to CSV: {mapping_output_path}")
        mapping_adata = AnnData(X=spot_to_cell_map.detach().cpu().numpy())
        mapping_adata.obs_names = adata_st.obs_names
        mapping_adata.var_names = adata_sc.obs_names
        anndata_to_csv(
            mapping_adata,
            mapping_output_path,
            top_left_label="Mapping",
            format_func=fmt_nonzero_4,
            uppercase_var_names=True,
        )
        logger.info(f"Saved spot-to-cell mapping to {mapping_output_path}")

    # Step 4: Compute Z' out of the mapping (expected gene expression per spot, scRNA data weighted by mapping)
    adata_prediction_prob = compute_gene_expression_prediction(
        spot_to_cell_map,
        cell_to_cell_type,
        adata_sc,
        adata_st,
        False,
        TORCH_DEVICE,
        training_config["use_cm"],
    )

    # Step 5 (optional): Export predicted_spot_expressions to CSV
    # - Rows: Genes
    # - Columns: Spots
    if output_path is not None:
        logger.info(f"Write result GEP to CSV: {output_path}")
        anndata_to_csv(
            adata_prediction_prob,
            output_path,
            top_left_label="GEP",
            format_func=fmt_nonzero_4,
            uppercase_var_names=True,
        )
        logger.info(f"Saved result GEP to {output_path}")
    else:
        logger.debug("No output path provided, skipping CSV export.")

    # Step 6 (optional): Apply one-hot encoding to mapping & repeat steps 4 & 5
    adata_prediction_det = None
    if mapping_config["deterministic"]:
        logger.info("Apply deterministic mapping & compute prediction with one-hot encoded mapping")

        adata_prediction_det = compute_gene_expression_prediction(
            spot_to_cell_map,
            cell_to_cell_type,
            adata_sc,
            adata_st,
            True,
            TORCH_DEVICE,
            training_config["use_cm"],
        )

        if output_path is not None:

            # Edit output_path: append "_deterministic" before file extension
            output_path_deterministic = output_path.with_name(output_path.stem + "_deterministic" + output_path.suffix)

            logger.info(f"Write result GEP to CSV: {output_path_deterministic}")
            anndata_to_csv(
                adata_prediction_det,
                output_path_deterministic,
                top_left_label="GEP",
                format_func=fmt_nonzero_4,
                uppercase_var_names=True,
            )
            logger.info(f"Saved result GEP to {output_path_deterministic}")
        else:
            logger.debug("No output path provided, skipping CSV export.")

    # Step 7: Return result
    return adata_prediction_prob, adata_prediction_det, losses_after_last_epoch


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Run AlternativeIdea alignment on a dataset folder")
    parser.add_argument('-d', '--dataset', dest='dataset', type=Path, help='Path to dataset folder')
    parser.add_argument('-c', '--config', dest='config', type=Path, help='Path to config.yaml')
    parser.add_argument('-o', '--output_path', dest='output_path', type=Path, required=False, default=None, help='Path where to store result to')
    parser.add_argument('-mo', '--mapping_output_path', type=Path, required=False, default=None, help='Path where to store mapping CSV to')
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
        args.mapping_output_path,
        verbose_logging=(args.logging == "verbose"),
    )
