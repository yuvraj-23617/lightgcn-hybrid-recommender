"""
lightgcn.py
===========
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
He et al., SIGIR 2020  —  faithful re-implementation from scratch.

Key design decisions (matching the paper):
  - NO feature-transformation weight matrices
  - NO non-linear activation functions
  - Symmetric normalisation coefficient only
  - Final embedding = mean of embeddings across all layers (0 … K)
  - Prediction score = inner product of final user & item embeddings
  - Training objective = BPR loss with L2 regularisation on layer-0 embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    Parameters
    ----------
    n_users     : number of users
    n_items     : number of items
    emb_dim     : embedding dimension d
    n_layers    : number of graph convolution layers K
    norm_adj    : pre-computed normalised adjacency sparse tensor
                  shape (n_users + n_items, n_users + n_items)
    lambda_reg  : L2 regularisation weight λ
    alpha       : layer combination weights.  None → uniform mean (1/K+1 each).
                  Otherwise pass a list of K+1 floats (must sum to 1).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_layers: int,
        norm_adj: torch.Tensor,
        lambda_reg: float = 1e-4,
        alpha: list[float] | None = None,
    ):
        super().__init__()
        self.n_users   = n_users
        self.n_items   = n_items
        self.emb_dim   = emb_dim
        self.n_layers  = n_layers
        self.norm_adj  = norm_adj           # kept as sparse COO
        self.lambda_reg = lambda_reg

        # Layer combination weights α_k
        if alpha is None:
            # Uniform: α_k = 1 / (K + 1) for all k
            self.alpha = None                # handled in forward()
        else:
            assert len(alpha) == n_layers + 1, "alpha must have n_layers+1 elements"
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

        # ------------------------------------------------------------------ #
        #  Layer-0 embeddings  E^(0)  — the ONLY learnable parameters        #
        # ------------------------------------------------------------------ #
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        self._init_weights()

    # ---------------------------------------------------------------------- #
    #  Weight initialisation                                                  #
    # ---------------------------------------------------------------------- #
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def init_semantic(self, semantic_item_emb: torch.Tensor):
        """
        Replace layer-0 item embeddings with pre-computed semantic vectors
        (e.g., from Sentence-BERT).  Tensor shape: (n_items, emb_dim).
        """
        assert semantic_item_emb.shape == (self.n_items, self.emb_dim), (
            f"Expected shape ({self.n_items}, {self.emb_dim}), "
            f"got {tuple(semantic_item_emb.shape)}"
        )
        with torch.no_grad():
            self.item_embedding.weight.copy_(semantic_item_emb)
        print("[LightGCN] Item embeddings initialised with semantic vectors.")

    # ---------------------------------------------------------------------- #
    #  Graph convolution                                                      #
    # ---------------------------------------------------------------------- #
    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run K layers of light graph convolution.

        Returns all_user_emb, all_item_emb — both shaped (n, emb_dim),
        already combined across layers.
        """
        device = self.user_embedding.weight.device

        # Concatenate user + item embeddings → E^(0)  shape: (n_users+n_items, d)
        E0 = torch.cat([self.user_embedding.weight,
                        self.item_embedding.weight], dim=0)

        all_layer_embs = [E0]                   # list of tensors, one per layer
        E_cur = E0

        norm_adj = self.norm_adj.to(device)

        for _ in range(self.n_layers):
            # E^(k) = Ã · E^(k-1)   (sparse × dense matrix multiply)
            E_cur = torch.sparse.mm(norm_adj, E_cur)
            all_layer_embs.append(E_cur)

        # Layer combination: mean (or weighted sum)
        stacked = torch.stack(all_layer_embs, dim=0)    # (K+1, n_users+n_items, d)

        if self.alpha is None:
            E_final = stacked.mean(dim=0)               # uniform mean
        else:
            w = self.alpha.to(device).view(-1, 1, 1)    # (K+1, 1, 1)
            E_final = (w * stacked).sum(dim=0)

        user_final = E_final[:self.n_users]
        item_final = E_final[self.n_users:]
        return user_final, item_final

    # ---------------------------------------------------------------------- #
    #  Forward pass                                                           #
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        users: torch.Tensor,        # (B,)
        pos_items: torch.Tensor,    # (B,)
        neg_items: torch.Tensor,    # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        bpr_loss   : scalar
        reg_loss   : scalar  (L2 on layer-0 embeddings only)
        """
        user_emb_final, item_emb_final = self._propagate()

        u  = user_emb_final[users]          # (B, d)
        pi = item_emb_final[pos_items]      # (B, d)
        ni = item_emb_final[neg_items]      # (B, d)

        # BPR loss
        pos_scores = (u * pi).sum(dim=1)    # (B,)
        neg_scores = (u * ni).sum(dim=1)    # (B,)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularisation on layer-0 embeddings
        reg_loss = (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / len(users)

        return bpr_loss, reg_loss

    # ---------------------------------------------------------------------- #
    #  Inference helpers                                                      #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def get_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final user and item embeddings (after layer combination)."""
        self.eval()
        return self._propagate()

    @torch.no_grad()
    def recommend(
        self,
        user_idx: int,
        exclude_items: set[int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Return top-K (item_idx, score) pairs for a single user,
        excluding already-seen items.
        """
        user_emb, item_emb = self.get_all_embeddings()
        u = user_emb[user_idx]                          # (d,)
        scores = (item_emb @ u).cpu().numpy()           # (n_items,)

        if exclude_items:
            for idx in exclude_items:
                scores[idx] = -1e9

        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
