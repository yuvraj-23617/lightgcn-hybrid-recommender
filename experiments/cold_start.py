"""
cold_start.py
=============
Evaluates and compares LightGCN (random init) vs LightGCN+SBERT
on cold-start users (≤5 training interactions).

Usage
-----
python experiments/cold_start.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, torch, numpy as np
from src.data_loader import ML1MDataset
from src.graph import build_norm_adj
from src.model.lightgcn import LightGCN
from src.evaluate import evaluate, evaluate_cold_start, print_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ML1MDataset(seed=42)
    norm_adj, _ = build_norm_adj(dataset.train_df, dataset.n_users, dataset.n_items, device)
    test_dict   = dataset.get_test_negatives()

    results = {}

    for tag, ckpt in [("LightGCN", "checkpoints/lightgcn_best.pt"),
                      ("LightGCN+SBERT", "checkpoints/lightgcn_sbert_best.pt")]:
        if not os.path.exists(ckpt):
            print(f"[cold_start] Checkpoint not found: {ckpt}  — skipping {tag}")
            continue

        model = LightGCN(dataset.n_users, dataset.n_items, 64, 3, norm_adj).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # Full test metrics
        full_metrics = evaluate(model, test_dict, dataset.n_users, dataset.n_items, device)
        print_metrics(full_metrics, prefix=f"[{tag}] FULL")

        # Cold-start metrics (≤5 interactions)
        for max_int in [5, 10]:
            cs_metrics = evaluate_cold_start(
                model, test_dict, dataset.user_pos_items, max_interactions=max_int, device=device
            )
            print_metrics(cs_metrics, prefix=f"[{tag}] COLD(≤{max_int})")

        results[tag] = {"full": full_metrics}

    os.makedirs("results", exist_ok=True)
    with open("results/cold_start_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[cold_start] Saved → results/cold_start_results.json")


if __name__ == "__main__":
    main()
