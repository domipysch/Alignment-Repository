import torch
import torch.nn as nn
from torch import Tensor
import logging
logger = logging.getLogger(__name__)


class AlternativeIdeaLoss(nn.Module):
    def __init__(
        self,
        lambda_rec_spot: float = 1.0,
        lambda_rec_state: float = 1.0,
        lambda_clust: float = 1.0,
        lambda_state_entropy: float = 1.0,
        lambda_spot_entropy: float = 1.0,
        eps: float = 1e-8
    ):
        super(AlternativeIdeaLoss, self).__init__()
        self.lambda_rec_spot = lambda_rec_spot
        self.lambda_rec_state = lambda_rec_state
        self.lambda_clust = lambda_clust
        self.lambda_state_entropy = lambda_state_entropy
        self.lambda_spot_entropy = lambda_spot_entropy
        self.eps = eps
        logger.debug("AlternativeIdeaLoss initialized")

    def get_rec_spot_loss(self, A: Tensor, X_shared: Tensor, Z_shared: Tensor) -> Tensor:
        """
        Eq (9): Scale-invariant reconstruction loss (Z' vs Z on shared genes).

        Args:
            A: Alignment matrix (S x C)
            X_shared: scRNA-seq reference restricted to shared genes (C x G_shared)
            Z_shared: Empirical HSO data restricted to shared genes (S x G_shared)
        """
        # --- Safety Check ---
        # Ensure that the number of genes (columns) matches
        if X_shared.shape[1] != Z_shared.shape[1]:
            raise ValueError(
                f"Gene dimension mismatch! X has {X_shared.shape[1]} genes, "
                f"but Z_marker has {Z_shared.shape[1]} genes."
            )

        # 1. Create Z_prime (The augmented spot data)
        # (S x C) @ (C x G_shared) -> (S x G_shared)
        Z_prime = torch.matmul(A, X_shared)

        # 2. Compute Cosine Similarity Components
        # Dot product across the gene dimension for each spot
        dot_product = torch.sum(Z_shared * Z_prime, dim=1)  # (S,)

        # L2 Norms for each spot
        norm_Z = torch.norm(Z_shared, p=2, dim=1)  # (S,)
        norm_Z_prime = torch.norm(Z_prime, p=2, dim=1)  # (S,)

        # 3. Calculate Scale-Invariant Loss
        # We add eps to the denominator for numerical stability
        cosine_sim = dot_product / (norm_Z * norm_Z_prime + self.eps)

        # Clip to range [-1, 1] to avoid nan due to float precision before distance calc
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)

        # Distance = sqrt(1 - similarity)
        # Or without sqrt?
        loss_per_spot = torch.sqrt(1.0 - cosine_sim)  # (S,)

        return torch.mean(loss_per_spot)

    def get_rec_state_loss(self, M_reconstructed: Tensor, A: Tensor, B: Tensor, X_sc: Tensor) -> Tensor:
        """
        Eq (10): Weighted reconstruction loss on cell state expression (M (=B^T x X) vs M_rec).

        Args:
            M_reconstructed: (K x G_sc) - The output of the decoder.
            A: (S x C) - Spot-to-cell assignment.
            B: (C x K) - Cell-to-state assignment.
            X_sc: (C x G_sc) - Reference scRNA-seq data.
        """
        # 1. Compute Matrix C (Spot -> State Mapping)
        # C = A @ B -> (S x K)
        C_mat = torch.matmul(A, B)

        # 2. Compute q (Spot state proportion vector)
        # q_k = |S|^-1 * sum over spots
        q = torch.mean(C_mat, dim=0)  # Shape: (K,)

        # 3. Compute empirical M from the reference data
        # M = B.T @ X_sc -> (K x C) @ (C x G_sc) = (K x G_sc)
        # This is what the states 'actually' look like given current assignments
        M_target = torch.matmul(B.t(), X_sc)

        # 4. Compute Squared L2 Norm per state
        # (M_rec - M_target)^2 summed over all genes G_sc
        # Result is a vector of K errors
        state_errors = torch.linalg.norm(M_reconstructed - M_target, ord=2, dim=1) ** 2  # (K,)

        # 5. Apply weights q_k and sum
        # This gives the total state reconstruction loss
        weighted_loss = torch.sum(q * state_errors)

        return weighted_loss

    def get_clust_loss(self, A: Tensor, h: Tensor, B: Tensor, F: Tensor) -> Tensor:
        """
        Eq (11): Cell clustering loss in embedding space.

        Links the learned cell state embeddings F to the cell embeddings E
        obtained from the spot embeddings H and the spot-to-cell assignment A.

        Args:
            A: (S x C) - Spot to cell mapping.
            h: (S x d) - Spatial embeddings from the GCN encoder.
            B: (C x K) - Cell to cell state mapping.
            F: (K x d) - Learned cell state embeddings.
        """
        # 1. E = A.T @ H - (C x d)
        E = torch.matmul(A.t(), h)

        # 2. Expected = B @ F - (C x d)
        E_expected = torch.matmul(B, F)

        # 3. Squared L2 Norm per cell (C,)
        # Note: we sum the squares over the d-dimension (dim=1)
        cell_errors = torch.linalg.norm(E - E_expected, ord=2, dim=1) ** 2

        # 4. Mean over the C-dimension
        return torch.mean(cell_errors)

    def get_state_entropy_loss(self, B: Tensor) -> Tensor:
        """
        Eq (12): Cell state entropy loss.

        Prioritizes the distribution of global cell-to-state assignments.

        Args:
            B: Cell-State assignment matrix (C x K).
        """
        # 1. Calculate p: The fraction of cells mapped to each state k
        # p is the mean of B across the cell dimension (dim=0)
        # Shape: (K,)
        p = torch.mean(B, dim=0)

        # 2. Compute Entropy: -sum(p * log(p + eps))
        # eps is added for numerical stability to avoid log(0)
        return -torch.sum(p * torch.log(p + self.eps))

    def get_spot_entropy_loss(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Eq (13): Spot state entropy loss.

        Prioritizes confident spot-to-cell-state assignments by minimizing
        the entropy of the derived spot-state matrix C.

        Args:
            A: Alignment matrix (S x C).
            B: Cell-State assignment matrix (C x K).
        """
        # 1. Compute C (S x K): Spot-to-State assignment matrix
        C = torch.matmul(A, B)

        # 2. Compute Entropy per spot
        # Formula: -sum_k(C_sk * log(C_sk + eps))
        # We perform the sum across the state dimension (dim=1)
        # Result shape: (S,)
        spot_entropies = -torch.sum(C * torch.log(C + self.eps), dim=1)

        # 3. Return the mean across all spots multiplied by lambda
        return torch.mean(spot_entropies)

    def forward(
        self,
        A: Tensor,
        B: Tensor,
        h: Tensor,
        M_rec: Tensor,
        F: Tensor,
        X: Tensor,
        X_shared: Tensor,
        Z_shared: Tensor
    ) -> dict[str, Tensor]:
        """
        Args:
            A: Spot to Cell mapping (S x C)
            B: Cell to cell state mapping (C x K)
            h: Spot embeddings (S x d)
            M_rec: Reconstructed state expression (K x g_sc)
            F: Cell state embeddings (K x d)
            X: Full scRNA-seq ref (C x g_sc)
            X_shared: scRNA-seq ref on shared genes (C x g_shared)
            Z_shared: Spatial data on shared genes (S x g_shared)
        """
        # 1. Individual terms (unweighted)
        l_rec_spot = self.get_rec_spot_loss(A, X_shared, Z_shared)
        l_rec_state = self.get_rec_state_loss(M_rec, A, B, X)
        l_clust = self.get_clust_loss(A, h, B, F)
        l_state_entropy = self.get_state_entropy_loss(B)
        l_spot_entropy = self.get_spot_entropy_loss(A, B)

        # 2. Weighted total
        total_loss = (
            self.lambda_rec_spot * l_rec_spot +
            self.lambda_rec_state * l_rec_state +
            self.lambda_clust * l_clust +
            self.lambda_state_entropy * l_state_entropy +
            self.lambda_spot_entropy * l_spot_entropy
        )

        return {
            "loss": total_loss,
            "rec_spot": l_rec_spot,
            "rec_state": l_rec_state,
            "clust": l_clust,
            "state_entropy": l_state_entropy,
            "spot_entropy": l_spot_entropy
        }

