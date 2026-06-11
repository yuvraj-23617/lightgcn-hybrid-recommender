"""
train_baselines.py
==================
Train all baseline models (MF, BPR-MF, NCF, NGCF) on MovieLens-1M
and record metrics for comparison with LightGCN.

Usage
-----
python experiments/train_baselines.py --model all
python experiments/train_baselines.py --model mf
python experiments/train_baselines.py --model ncf --emb_dim 64
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import time
import torch
import numpy as np
from torch.optim import Adam

from src.data_loader import ML1MDataset
from src.graph import build_norm_adj
from src.model.mf   import MatrixFactorization
from src.model.ncf  import NCF
from src.model.ngcf import NGCF
from src.evaluate   import evaluate, print_metrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="all", choices=["all", "mf", "ncf", "ngcf"])
    p.add_argument("--data_dir",   default="data/ml-1m")
    p.add_argument("--emb_dim",    type=int,   default=64)
    p.add_argument("--n_layers",   type=int,   default=3,    help="NGCF only")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=2048)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--eval_every", type=int,   default=5)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def train_model(model, model_name, dataset, val_dict, test_dict, args, device):
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_ndcg   = 0.0
    patience_cnt = 0
    history      = []
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{model_name}_best.pt")

    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}  |  emb_dim={args.emb_dim}  lr={args.lr}")
    print(f"{'='*55}\n")

    n_batches = max(1, len(dataset.train_df) // args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for _ in range(n_batches):
            users, pos_items, neg_items = dataset.sample_bpr_batch(args.batch_size)
            users     = torch.LongTensor(users).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)
            neg_items = torch.LongTensor(neg_items).to(device)

            bpr_loss, reg_loss = model(users, pos_items, neg_items)
            loss = bpr_loss + args.lambda_reg * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_dict, dataset.n_users, dataset.n_items, device)
            ndcg10 = val_metrics.get("NDCG@10", 0.0)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:>4d}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s", end="  ")
            print_metrics(val_metrics)
            history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                patience_cnt = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [BEST] NDCG@10={best_ndcg:.4f}")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"\n[{model_name}] Early stop at epoch {epoch}")
                    break

    # Test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_dict, dataset.n_users, dataset.n_items, device)
    print_metrics(test_metrics, prefix=f"\n[TEST] {model_name}")

    return {"model": model_name, "config": vars(args), "test_metrics": test_metrics, "history": history}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baselines] Device: {device}")

    dataset  = ML1MDataset(data_dir=args.data_dir, seed=args.seed)
    norm_adj, _ = build_norm_adj(dataset.train_df, dataset.n_users, dataset.n_items, device)
    val_dict    = dataset.get_val_negatives()
    test_dict   = dataset.get_test_negatives()

    models_to_run = ["mf", "ncf", "ngcf"] if args.model == "all" else [args.model]
    all_results   = {}

    for name in models_to_run:
        if name == "mf":
            model = MatrixFactorization(dataset.n_users, dataset.n_items, args.emb_dim, args.lambda_reg).to(device)
        elif name == "ncf":
            model = NCF(dataset.n_users, dataset.n_items, args.emb_dim, lambda_reg=args.lambda_reg).to(device)
        elif name == "ngcf":
            model = NGCF(dataset.n_users, dataset.n_items, args.emb_dim, args.n_layers, norm_adj, lambda_reg=args.lambda_reg).to(device)

        result = train_model(model, name, dataset, val_dict, test_dict, args, device)
        all_results[name] = result

    # Save combined results
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[baselines] Saved -> results/baseline_results.json")

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<12} {'HR@10':>8} {'NDCG@10':>10} {'HR@20':>8} {'NDCG@20':>10}")
    print("-" * 60)
    for name, res in all_results.items():
        m = res["test_metrics"]
        print(f"{name:<12} {m.get('HR@10',0):>8.4f} {m.get('NDCG@10',0):>10.4f} {m.get('HR@20',0):>8.4f} {m.get('NDCG@20',0):>10.4f}")


if __name__ == "__main__":
    main()
