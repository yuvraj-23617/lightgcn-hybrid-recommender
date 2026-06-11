"""
ablations.py
============
Runs ablation studies for LightGCN:
  1. Number of propagation layers  K ∈ {1, 2, 3, 4}
  2. Embedding dimension           d ∈ {16, 32, 64, 128}
  3. Layer combination weights     uniform vs. learned α_k

Usage
-----
python experiments/ablations.py --study layers
python experiments/ablations.py --study dim
python experiments/ablations.py --study all
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, json, time
import torch, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam

from src.data_loader import ML1MDataset
from src.graph import build_norm_adj
from src.model.lightgcn import LightGCN
from src.evaluate import evaluate


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="all", choices=["layers", "dim", "all"])
    p.add_argument("--data_dir",   default="data/ml-1m")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--patience",   type=int, default=10)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def quick_train(model, dataset, val_dict, args, device, tag):
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_ndcg, best_state, patience_cnt = 0.0, None, 0
    n_batches = max(1, len(dataset.train_df) // args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for _ in range(n_batches):
            u, pi, ni = dataset.sample_bpr_batch(args.batch_size)
            u  = torch.LongTensor(u).to(device)
            pi = torch.LongTensor(pi).to(device)
            ni = torch.LongTensor(ni).to(device)
            bpr, reg = model(u, pi, ni)
            loss = bpr + args.lambda_reg * reg
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % args.eval_every == 0:
            val_m = evaluate(model, val_dict, dataset.n_users, dataset.n_items, device)
            ndcg10 = val_m.get("NDCG@10", 0.0)
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                import copy; best_state = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    break

    model.load_state_dict(best_state)
    return model


def run_layer_ablation(dataset, norm_adj, val_dict, test_dict, args, device):
    Ks = [1, 2, 3, 4]
    results = {}
    for K in Ks:
        print(f"\n── K={K} ──")
        model = LightGCN(dataset.n_users, dataset.n_items, 64, K, norm_adj, args.lambda_reg).to(device)
        model = quick_train(model, dataset, val_dict, args, device, f"K{K}")
        m = evaluate(model, test_dict, dataset.n_users, dataset.n_items, device)
        results[K] = m
        print(f"   K={K}  NDCG@10={m['NDCG@10']:.4f}  HR@10={m['HR@10']:.4f}")
    return results


def run_dim_ablation(dataset, norm_adj, val_dict, test_dict, args, device):
    dims = [16, 32, 64, 128]
    results = {}
    for d in dims:
        print(f"\n── d={d} ──")
        model = LightGCN(dataset.n_users, dataset.n_items, d, 3, norm_adj, args.lambda_reg).to(device)
        model = quick_train(model, dataset, val_dict, args, device, f"d{d}")
        m = evaluate(model, test_dict, dataset.n_users, dataset.n_items, device)
        results[d] = m
        print(f"   d={d}  NDCG@10={m['NDCG@10']:.4f}  HR@10={m['HR@10']:.4f}")
    return results


def plot_ablation(results: dict, x_label: str, metric: str = "NDCG@10", title: str = ""):
    keys   = list(results.keys())
    values = [results[k].get(metric, 0) for k in keys]
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=keys, y=values, marker="o", linewidth=2.5)
    plt.title(title or f"{metric} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fname = f"results/ablation_{x_label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[ablation] Plot saved → {fname}")


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset   = ML1MDataset(data_dir=args.data_dir, seed=args.seed)
    norm_adj, _ = build_norm_adj(dataset.train_df, dataset.n_users, dataset.n_items, device)
    val_dict    = dataset.get_val_negatives()
    test_dict   = dataset.get_test_negatives()

    all_results = {}

    if args.study in ("layers", "all"):
        print("\n━━━ Ablation: Number of Layers ━━━")
        layer_results = run_layer_ablation(dataset, norm_adj, val_dict, test_dict, args, device)
        all_results["layers"] = {str(k): v for k, v in layer_results.items()}
        plot_ablation(layer_results, "K (layers)", "NDCG@10", "NDCG@10 vs. Number of Layers")

    if args.study in ("dim", "all"):
        print("\n━━━ Ablation: Embedding Dimension ━━━")
        dim_results = run_dim_ablation(dataset, norm_adj, val_dict, test_dict, args, device)
        all_results["dim"] = {str(k): v for k, v in dim_results.items()}
        plot_ablation(dim_results, "Embedding Dim", "NDCG@10", "NDCG@10 vs. Embedding Dimension")

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[ablation] Saved → results/ablation_results.json")


if __name__ == "__main__":
    main()
