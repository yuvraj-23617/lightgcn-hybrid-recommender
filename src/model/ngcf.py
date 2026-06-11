"""
ngcf.py  —  Neural Graph Collaborative Filtering baseline
==========================================================
Wang et al., SIGIR 2019
GCN with learnable weight matrices and LeakyReLU activations per layer.
This is the predecessor of LightGCN; it is intentionally heavier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_layers: int,
        norm_adj: torch.Tensor,
        dropout: float = 0.1,
        lambda_reg: float = 1e-4,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.lambda_reg = lambda_reg
        self.dropout = dropout

        # Layer-0 embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # Weight matrices W1, W2 per layer  (2 matrices per layer as in the paper)
        self.W1 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for w in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(w.weight)

    def _propagate(self):
        device = self.user_embedding.weight.device
        norm_adj = self.norm_adj.to(device)

        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)  # (N, d)
        all_embs = [E]

        for k in range(self.n_layers):
            # Neighborhood aggregation
            agg = torch.sparse.mm(norm_adj, E)                              # Ã·E
            # Element-wise product term (interaction between E and aggregated E)
            side = torch.sparse.mm(norm_adj, E * E)                        # Ã·(E⊙E)  — approx interaction

            # NGCF update: LeakyReLU(W1·agg + W2·side)
            E = F.leaky_relu(self.W1[k](agg) + self.W2[k](side), negative_slope=0.2)
            # Dropout for regularisation
            E = F.dropout(E, p=self.dropout, training=self.training)
            all_embs.append(E)

        # Concatenate all layer outputs (as in original NGCF paper)
        E_final = torch.cat(all_embs, dim=1)    # (N, d*(K+1))
        user_final = E_final[:self.n_users]
        item_final = E_final[self.n_users:]
        return user_final, item_final

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        user_emb, item_emb = self._propagate()

        u  = user_emb[users]
        pi = item_emb[pos_items]
        ni = item_emb[neg_items]

        pos_scores = (u * pi).sum(dim=1)
        neg_scores = (u * ni).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / len(users)

        return bpr_loss, reg_loss

    @torch.no_grad()
    def get_all_embeddings(self):
        self.eval()
        return self._propagate()

    @torch.no_grad()
    def recommend(self, user_idx: int, exclude_items: set = None, top_k: int = 10):
        user_emb, item_emb = self.get_all_embeddings()
        u = user_emb[user_idx]
        scores = (item_emb @ u).cpu().numpy()
        if exclude_items:
            for idx in exclude_items:
                scores[idx] = -1e9
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
