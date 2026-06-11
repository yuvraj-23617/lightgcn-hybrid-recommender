"""
ncf.py  —  Neural Collaborative Filtering baseline
===================================================
He et al., WWW 2017
Combines GMF (element-wise product) + MLP, fused via NeuMF output layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        mlp_layers: list[int] = None,
        dropout: float = 0.2,
        lambda_reg: float = 1e-4,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_reg = lambda_reg

        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        # GMF embeddings
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)

        # MLP embeddings
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # MLP tower
        mlp_input_dim = emb_dim * 2
        layers = []
        prev = mlp_input_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(prev, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = out_dim
        self.mlp = nn.Sequential(*layers)

        # NeuMF output
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def _score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in  = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1)
        mlp_out = self.mlp(mlp_in)
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output(combined).squeeze(1)

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        pos_scores = self._score(users, pos_items)
        neg_scores = self._score(users, neg_items)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (
            self.gmf_user(users).norm(2).pow(2) +
            self.gmf_item(pos_items).norm(2).pow(2) +
            self.gmf_item(neg_items).norm(2).pow(2) +
            self.mlp_user(users).norm(2).pow(2) +
            self.mlp_item(pos_items).norm(2).pow(2) +
            self.mlp_item(neg_items).norm(2).pow(2)
        ) / len(users)

        return bpr_loss, reg_loss

    @torch.no_grad()
    def get_all_embeddings(self):
        """
        For NCF, 'embeddings' are the GMF embeddings (used for dot-product evaluation).
        We return the GMF embedding tables so the evaluate.py interface works.
        """
        return self.gmf_user.weight, self.gmf_item.weight

    @torch.no_grad()
    def recommend(self, user_idx: int, exclude_items: set = None, top_k: int = 10):
        self.eval()
        n_items = self.gmf_item.weight.shape[0]
        users_t = torch.tensor([user_idx] * n_items)
        items_t = torch.arange(n_items)
        scores = self._score(users_t, items_t).cpu().numpy()
        if exclude_items:
            for idx in exclude_items:
                scores[idx] = -1e9
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
