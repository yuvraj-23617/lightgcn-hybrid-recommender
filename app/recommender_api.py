"""
recommender_api.py
==================
Python API that wraps the trained LightGCN model for use in the Streamlit app.
Provides simple, high-level methods for getting recommendations.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from src.data_loader import ML1MDataset
from src.graph import build_norm_adj
from src.model.lightgcn import LightGCN


class RecommenderAPI:
    """
    High-level recommendation API wrapping a trained LightGCN model.

    Parameters
    ----------
    checkpoint : path to .pt checkpoint file (default: best available)
    emb_dim    : embedding dimension used during training
    n_layers   : number of graph convolution layers
    data_dir   : path to MovieLens-1M data
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/lightgcn_best.pt",
        emb_dim:    int = 64,
        n_layers:   int = 3,
        data_dir:   str = "data/ml-1m",
        device:     str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[RecommenderAPI] Loading dataset …")
        self.dataset = ML1MDataset(data_dir=data_dir)

        print(f"[RecommenderAPI] Building graph …")
        norm_adj, _ = build_norm_adj(
            self.dataset.train_df,
            self.dataset.n_users,
            self.dataset.n_items,
            self.device,
        )

        print(f"[RecommenderAPI] Loading model from {checkpoint} …")
        self.model = LightGCN(
            n_users    = self.dataset.n_users,
            n_items    = self.dataset.n_items,
            emb_dim    = emb_dim,
            n_layers   = n_layers,
            norm_adj   = norm_adj,
        ).to(self.device)

        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            print(f"[RecommenderAPI] Checkpoint loaded ✓")
        else:
            print(f"[RecommenderAPI] ⚠ Checkpoint not found — using untrained model (for demo).")

        self.model.eval()

        # Pre-compute all embeddings once
        print("[RecommenderAPI] Pre-computing embeddings …")
        with torch.no_grad():
            self.user_emb, self.item_emb = self.model.get_all_embeddings()
        print("[RecommenderAPI] Ready ✓")

    # ------------------------------------------------------------------ #
    #  Core recommendation methods                                        #
    # ------------------------------------------------------------------ #

    def get_recommendations(self, user_id: int, k: int = 10) -> list[dict]:
        """
        Get top-K recommendations for a given user index.

        Returns list of dicts: {item_idx, title, genres, score}
        """
        if user_id < 0 or user_id >= self.dataset.n_users:
            raise ValueError(f"user_id must be in [0, {self.dataset.n_users-1}], got {user_id}")

        exclude = self.dataset.user_all_pos.get(user_id, set())

        u = self.user_emb[user_id]
        scores = (self.item_emb @ u).cpu().numpy()

        for idx in exclude:
            if idx < len(scores):
                scores[idx] = -1e9

        top_indices = scores.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            results.append({
                "item_idx": int(idx),
                "title":    self.dataset.get_movie_title(int(idx)),
                "genres":   self.dataset.get_movie_genres(int(idx)),
                "score":    float(scores[idx]),
            })
        return results

    def get_user_history(self, user_id: int, k: int = 10) -> list[dict]:
        """Return the user's most recent training interactions."""
        if user_id not in self.dataset.user_pos_items:
            return []
        items = list(self.dataset.user_pos_items[user_id])
        return [
            {
                "item_idx": i,
                "title":    self.dataset.get_movie_title(i),
                "genres":   self.dataset.get_movie_genres(i),
            }
            for i in items[:k]
        ]

    def get_similar_items(self, item_idx: int, k: int = 10) -> list[dict]:
        """Return k most similar items to a given item (cosine similarity in embedding space)."""
        if item_idx < 0 or item_idx >= self.dataset.n_items:
            raise ValueError(f"item_idx out of range")

        query = self.item_emb[item_idx]          # (d,)
        # Cosine similarity
        norms  = self.item_emb.norm(dim=1)
        q_norm = query.norm()
        sims   = (self.item_emb @ query) / (norms * q_norm + 1e-8)
        sims   = sims.cpu().numpy()
        sims[item_idx] = -1e9                    # exclude self

        top_indices = sims.argsort()[::-1][:k]
        return [
            {
                "item_idx": int(i),
                "title":    self.dataset.get_movie_title(int(i)),
                "genres":   self.dataset.get_movie_genres(int(i)),
                "similarity": float(sims[i]),
            }
            for i in top_indices
        ]

    def get_content_based(self, genres: list[str], k: int = 10) -> list[dict]:
        """
        Simple content-based fallback: find items matching any of the given genres.
        Returns k items sorted by their average item embedding norm
        (popular items tend to have higher-norm embeddings in BPR training).
        """
        scored = []
        movies = self.dataset.movies_df
        norms  = self.item_emb.norm(dim=1).cpu().numpy()

        for item_idx, row in movies.iterrows():
            item_genres = str(row.get("genres", "")).lower()
            if any(g.lower() in item_genres for g in genres):
                scored.append({
                    "item_idx": int(item_idx),
                    "title":    row.get("title", f"Movie {item_idx}"),
                    "genres":   row.get("genres", ""),
                    "score":    float(norms[item_idx]) if item_idx < len(norms) else 0.0,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    @property
    def n_users(self):
        return self.dataset.n_users

    @property
    def n_items(self):
        return self.dataset.n_items
