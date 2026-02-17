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
        eps: float = 1e-8,
        normalize_by_initial: bool = True,
        warmup_iters: int = 1,
        k: int = 20,
        use_cm: bool = False,
    ):
        super(AlternativeIdeaLoss, self).__init__()
        self.lambda_rec_spot = lambda_rec_spot
        self.lambda_rec_state = lambda_rec_state
        self.lambda_clust = lambda_clust
        self.lambda_state_entropy = lambda_state_entropy
        self.lambda_spot_entropy = lambda_spot_entropy
        self.eps = eps
        self.lnK = torch.log(torch.tensor(k, dtype=torch.float32))

        # Normalization / baseline settings for rec_state and clust
        self.normalize_by_initial = bool(normalize_by_initial)
        self.warmup_iters = int(warmup_iters) if warmup_iters >= 1 else 1

        # Buffers to hold baseline statistics (registered buffers move with module)
        # Initialize with NaN to indicate not-yet-set
        self.register_buffer("_init_rec_state", torch.tensor(float("nan")))
        self.register_buffer("_init_clust", torch.tensor(float("nan")))

        self.use_cm = use_cm

        # vielleicht ist das mit dem mean overengineered und wir brauchen einfach nur ersten wert. todo. check.
        # Accumulators used during warmup to compute the baseline averages
        self._acc_rec_state = 0.0
        self._acc_clust = 0.0
        self._forward_count = 0
        logger.debug("AlternativeIdeaLoss initialized")


    def get_rec_spot_loss(self, A: Tensor, B: Tensor, X_shared: Tensor, Z_shared: Tensor) -> Tensor:
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

        # 1. Create Z_prime (The augmented spot data reconstructed from the scRNA-seq reference)
        if self.use_cm:

            C = torch.matmul(A, B)  # (S x K)

            # Compute B_normalized: Normalize B^T by the number of cells assigned to each state to prevent scale issues
            B_normalized_sum_over_types_1 = B / (torch.sum(B, dim=0) + self.eps)  # (K x C)

            M = torch.matmul(B_normalized_sum_over_types_1.t(), X_shared)  # (K x G_shared)
            # M = torch.matmul(B.t(), X_shared)  # (K x G_shared)
            Z_prime = torch.matmul(C, M)  # (S x G_shared)

        else:
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
        # loss_per_spot = torch.sqrt(1.0 - cosine_sim)  # (S,)
        # todo. Or doch without sqrt?, gab nan-probleme mit
        loss_per_spot = torch.clamp(1.0 - cosine_sim, min=0.0)

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
        # weighted_loss = torch.sum(q * state_errors)
        # todo. check. war sum!
        weighted_loss = torch.mean(q * state_errors)

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
        l_rec_spot = self.get_rec_spot_loss(A, B, X_shared, Z_shared)
        l_rec_state = self.get_rec_state_loss(M_rec, A, B, X)
        l_clust = self.get_clust_loss(A, h, B, F)
        l_state_entropy = self.get_state_entropy_loss(B)
        l_spot_entropy = self.get_spot_entropy_loss(A, B)

        # 2. Optionally normalize rec_state and clust by their initial baseline
        # We use a warmup period: collect raw values for `warmup_iters` forwards
        l_rec_state_norm = l_rec_state
        l_clust_norm = l_clust

        if self.normalize_by_initial:
            # Accumulate raw scalar values during warmup
            try:
                raw_rs = float(l_rec_state.detach().cpu().item())
                raw_cl = float(l_clust.detach().cpu().item())
            except Exception:
                # defensive fallback
                raw_rs = float(l_rec_state.detach().cpu().numpy())
                raw_cl = float(l_clust.detach().cpu().numpy())

            if self._forward_count < self.warmup_iters:
                self._acc_rec_state += raw_rs
                self._acc_clust += raw_cl

            self._forward_count += 1

            # If warmup just finished, set initial baselines as the average across warmup
            if self._forward_count == self.warmup_iters:
                init_rs = max(self._acc_rec_state / float(self.warmup_iters), float(self.eps))
                init_cl = max(self._acc_clust / float(self.warmup_iters), float(self.eps))
                # store as tensors on the module (register_buffer created them)
                self._init_rec_state = torch.tensor(init_rs, device=l_rec_state.device)
                self._init_clust = torch.tensor(init_cl, device=l_clust.device)
                logger.info(f"Loss baselines set: init_rec_state={init_rs:.6g}, init_clust={init_cl:.6g}")

            # If baselines are available (not NaN), divide by baseline
            if not torch.isnan(self._init_rec_state):
                denom_rs = (self._init_rec_state + self.eps).to(l_rec_state.device)
                l_rec_state_norm = l_rec_state / denom_rs
            if not torch.isnan(self._init_clust):
                denom_cl = (self._init_clust + self.eps).to(l_clust.device)
                l_clust_norm = l_clust / denom_cl

        # 3. Normalize l_state_entropy and l_spot_entropy by their maximum possible values to keep them in a comparable range
        # Max entropy for both is log(K)
        l_state_entropy /= self.lnK
        l_spot_entropy /= self.lnK

        # 3. Weighted total (use normalized versions for rec_state and clust when available)
        total_loss = (
            self.lambda_rec_spot * l_rec_spot +
            self.lambda_rec_state * l_rec_state_norm +
            self.lambda_clust * l_clust_norm +
            self.lambda_state_entropy * l_state_entropy +
            self.lambda_spot_entropy * l_spot_entropy
        )

        # Return both the weighted normalized components (for training/plotting) and the raw-weighted ones (for debug)
        return {
            "loss": total_loss,
            "rec_spot": l_rec_spot,
            "rec_state": l_rec_state_norm,
            "clust": l_clust_norm,
            "state_entropy": l_state_entropy,
            "spot_entropy": l_spot_entropy,
        }
