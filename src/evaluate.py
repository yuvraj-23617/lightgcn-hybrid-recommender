"""
evaluate.py
===========
Evaluation metrics for top-K recommendation.

Metrics implemented:
  - Hit Rate @ K  (HR@K)        — was the positive item in the top-K list?
  - NDCG @ K                    — normalised discounted cumulative gain

Evaluation protocol (leave-one-out with sampled negatives):
  For each test user, rank the 1 positive item against 99 randomly sampled
  negative items (100 items total).  Compute HR@K and NDCG@K at K=10, 20.
"""

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-user metric helpers
# ---------------------------------------------------------------------------

def hit_rate(ranked_list: list[int], pos_item: int) -> float:
    """1.0 if pos_item is in ranked_list, else 0.0."""
    return 1.0 if pos_item in ranked_list else 0.0


def ndcg(ranked_list: list[int], pos_item: int) -> float:
    """NDCG for a single positive item."""
    if pos_item in ranked_list:
        rank = ranked_list.index(pos_item)          # 0-indexed
        return 1.0 / np.log2(rank + 2)              # +2 because log2(1)=0
    return 0.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    test_dict: dict,          # {user_idx: (pos_item, [neg_items])}
    n_users: int,
    n_items: int,
    device: torch.device,
    ks: list[int] = [10, 20],
    batch_size: int = 512,
) -> dict:
    """
    Evaluate a model on the test (or val) set.

    Parameters
    ----------
    model      : any model with a .get_all_embeddings() → (user_emb, item_emb)
    test_dict  : mapping from user_idx to (pos_item_idx, list_of_neg_item_idxs)
    ks         : list of K values to evaluate at

    Returns
    -------
    metrics : dict  e.g. {"HR@10": 0.73, "NDCG@10": 0.42, "HR@20": ..., "NDCG@20": ...}
    """
    model.eval()

    # Get full user & item embedding matrices
    user_emb, item_emb = model.get_all_embeddings()  # (n_users, d), (n_items, d)
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    results = {f"HR@{k}": [] for k in ks}
    results.update({f"NDCG@{k}": [] for k in ks})

    users = list(test_dict.keys())

    for start in range(0, len(users), batch_size):
        batch_users = users[start: start + batch_size]
        u_emb = user_emb[batch_users]                   # (B, d)

        for i, u in enumerate(batch_users):
            pos_item, neg_items = test_dict[u]
            candidate_items = [pos_item] + list(neg_items)

            i_emb = item_emb[candidate_items]           # (C, d)
            scores = (u_emb[i:i+1] * i_emb).sum(dim=1) # (C,)

            # Rank candidates (descending)
            order = torch.argsort(scores, descending=True).cpu().numpy()
            ranked_candidates = [candidate_items[idx] for idx in order]

            for k in ks:
                top_k = ranked_candidates[:k]
                results[f"HR@{k}"].append(hit_rate(top_k, pos_item))
                results[f"NDCG@{k}"].append(ndcg(top_k, pos_item))

    # Average
    return {key: float(np.mean(vals)) for key, vals in results.items()}


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print a metrics dict."""
    parts = [f"{k}: {v:.4f}" for k, v in sorted(metrics.items())]
    print(f"{prefix}  " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Cold-start evaluation helper
# ---------------------------------------------------------------------------

def evaluate_cold_start(
    model,
    test_dict: dict,
    user_pos_items: dict,        # {user_idx: set of train pos items}
    max_interactions: int = 5,
    device: torch.device = torch.device("cpu"),
    ks: list[int] = [10, 20],
) -> dict:
    """
    Evaluate only on users with ≤ max_interactions training interactions.
    Useful for measuring cold-start performance.
    """
    cold_users = {
        u: v for u, v in test_dict.items()
        if len(user_pos_items.get(u, set())) <= max_interactions
    }
    if not cold_users:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}
    print(f"[eval] Cold-start users (≤{max_interactions} interactions): {len(cold_users)}")
    return evaluate(model, cold_users, None, None, device, ks)
