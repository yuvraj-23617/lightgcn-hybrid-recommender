"""
mf.py  —  Standard Matrix Factorization baseline
=================================================
Uses MSE loss on explicit (or implicit) ratings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, lambda_reg: float = 1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_reg = lambda_reg

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        """BPR loss variant for MF (used for fair comparison)."""
        u  = self.user_embedding(users)
        pi = self.item_embedding(pos_items)
        ni = self.item_embedding(neg_items)

        pos_scores = (u * pi).sum(dim=1) + self.user_bias(users).squeeze() + self.item_bias(pos_items).squeeze()
        neg_scores = (u * ni).sum(dim=1) + self.user_bias(users).squeeze() + self.item_bias(neg_items).squeeze()

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg_loss = (
            self.user_embedding(users).norm(2).pow(2) +
            self.item_embedding(pos_items).norm(2).pow(2) +
            self.item_embedding(neg_items).norm(2).pow(2)
        ) / len(users)

        return bpr_loss, reg_loss

    @torch.no_grad()
    def get_all_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight

    @torch.no_grad()
    def recommend(self, user_idx: int, exclude_items: set = None, top_k: int = 10):
        u = self.user_embedding.weight[user_idx]
        scores = (self.item_embedding.weight @ u).cpu().numpy()
        if exclude_items:
            for idx in exclude_items:
                scores[idx] = -1e9
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
