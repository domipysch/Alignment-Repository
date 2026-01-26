import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import GCNConv
import logging
logger = logging.getLogger(__name__)


class AlternativeIdeaModel(nn.Module):
    def __init__(
        self,
        num_spots_st: int,
        num_cells_sc: int,
        g_st: int,
        g_sc: int,
        d: int = 32,
        k: int = 20,
        enc_hidden_dim: int = 64,
        dec_hidden_dim: int = 256,
        dropout_rate_decoder: float = 0.2,
    ):
        """
        Args:
            num_spots_st (S): Number of spots in Z
            num_cells_sc (C): Number of cells in X
            g_st: Number of genes in Z
            g_sc: Number of genes in X
            d: Dimension of the embedding space
            k: Upper bound on number of cell states
            dropout_rate_decoder: Dropout rate for regularization in the decoder
        """
        super(AlternativeIdeaModel, self).__init__()
        self.num_cells_sc = num_cells_sc

        # --- 1. ENCODER (GCN) ---
        # Processes spatial data while considering the neighborhood structure
        self.conv1 = GCNConv(g_st, enc_hidden_dim)
        self.conv2 = GCNConv(enc_hidden_dim, d)

        # --- 2. TRAINABLE MATRICES (The mapping to be optimized) ---
        # U: Spot to cell mapping (S x C)
        # This will become A after Softmax
        self.U = nn.Parameter(torch.randn(num_spots_st, num_cells_sc))

        # V: Cell to cell state mapping (C x K)
        # This will become B after Softmax
        self.V = nn.Parameter(torch.randn(num_cells_sc, k))

        # F: Cell state embedding (K x d)
        self.F = nn.Parameter(torch.randn(k, d))

        # --- 3. DECODER (MLP) ---
        # Reconstructs the full gene profile (G_sc genes) from the embedding space
        self.decoder = nn.Sequential(
            nn.Linear(d, dec_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate_decoder),
            nn.Linear(dec_hidden_dim, g_sc)
        )

    def forward(self, z: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            z: Spatial gene features
            edge_index: Graph connectivity

        Returns:
            A: Spot-Cell alignment matrix (S x C)
            B: Cell-State assignment matrix (C x K)
            h: Spatial embeddings (S x d)
            M_rec: Accumulated gene expression per state (K x g_sc)
            F: Cell state embeddings (K x d)
        """
        logger.debug("Forward pass")

        # A. Calculate Spot Embeddings (H)
        # This captures the spatial context of each spot
        h = self.conv1(z, edge_index).relu()
        h = self.conv2(h, edge_index)  # Output shape: S x d

        # B. Calculate Alignment Matrices
        # We use Softmax to ensure rows sum to 1.0 (probabilities/proportions)
        # A is the Spot -> Cell mapping (S X C)
        A = torch.softmax(self.U, dim=1)

        # B is the Cell -> State (C x K)
        B = torch.softmax(self.V, dim=1)

        # C. Calculate p (State Proportions)
        # p_k = (1/|C|) * sum_c(B_c,k)
        p = torch.mean(B, dim=0)  # Shape: (K)

        # D. Reconstruct Total Expression Matrix M
        # 1. Get average gene expression profiles from F
        # profiles shape: (K x g_sc)
        profiles = self.decoder(self.F)

        # 2. Scale profiles by total cell count per state
        # Total cells per state = |C| * p
        M_rec = profiles * (self.num_cells_sc * p.unsqueeze(1))

        return A, B, h, M_rec, self.F
