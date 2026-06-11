"""
train_lightgcn.py
=================
Training entry point for LightGCN (and LightGCN+SBERT variant).

Usage
-----
# Standard LightGCN
python experiments/train_lightgcn.py

# With Sentence-BERT semantic initialisation
python experiments/train_lightgcn.py --semantic

# Custom hyperparameters
python experiments/train_lightgcn.py --n_layers 3 --emb_dim 64 --lr 1e-3 --epochs 200
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Force UTF-8 output on Windows to avoid cp1252 crashes
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import time
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.data_loader import ML1MDataset
from src.graph import build_norm_adj
from src.model.lightgcn import LightGCN
from src.evaluate import evaluate, print_metrics
from src.semantic_encoder import encode_items


# ─────────────────────────────── CLI ─────────────────────────────────────── #

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/ml-1m")
    p.add_argument("--emb_dim",    type=int,   default=64)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=2048)
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--patience",   type=int,   default=15,   help="Early stopping patience")
    p.add_argument("--eval_every", type=int,   default=5)
    p.add_argument("--top_k",      type=int,   nargs="+",    default=[10, 20])
    p.add_argument("--semantic",   action="store_true",       help="Use SBERT item init")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--log_file",   default="results/train_log.txt", help="Progress log file")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def log(msg, log_path=None):
    """Print to stdout (flushed) and optionally write to a log file."""
    print(msg, flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ─────────────────────────────── Main ────────────────────────────────────── #

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = args.log_file   # shorthand
    # Clear old log
    if L:
        os.makedirs(os.path.dirname(L) or ".", exist_ok=True)
        open(L, "w").close()

    log(f"[train] Device: {device}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", L)

    # -- Data ------------------------------------------------------------- #
    dataset = ML1MDataset(data_dir=args.data_dir, seed=args.seed)
    norm_adj, _ = build_norm_adj(dataset.train_df, dataset.n_users, dataset.n_items, device)

    # -- Model ------------------------------------------------------------ #
    model = LightGCN(
        n_users    = dataset.n_users,
        n_items    = dataset.n_items,
        emb_dim    = args.emb_dim,
        n_layers   = args.n_layers,
        norm_adj   = norm_adj,
        lambda_reg = args.lambda_reg,
    ).to(device)

    # Optional: SBERT semantic initialisation
    if args.semantic:
        log("[train] Computing Sentence-BERT item embeddings...", L)
        sbert_emb = encode_items(dataset, emb_dim=args.emb_dim, device=device)
        model.init_semantic(sbert_emb)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # -- Training loop ---------------------------------------------------- #
    tag = "lightgcn_sbert" if args.semantic else "lightgcn"
    os.makedirs(args.save_dir, exist_ok=True)
    best_ndcg   = 0.0
    patience_cnt = 0
    history      = []

    # Pre-compute negatives ONCE before the loop
    log("[train] Pre-computing val negatives (one-time)...", L)
    val_dict  = dataset.get_val_negatives()
    log("[train] Pre-computing test negatives (one-time)...", L)
    test_dict = dataset.get_test_negatives()
    log("[train] Negatives ready. Starting training...\n", L)

    log(f"\n{'='*60}", L)
    log(f"  LightGCN  |  layers={args.n_layers}  dim={args.emb_dim}  lr={args.lr}  semantic={args.semantic}", L)
    log(f"{'='*60}\n", L)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = max(1, len(dataset.train_df) // args.batch_size)
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

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_dict, dataset.n_users, dataset.n_items, device, args.top_k)
            ndcg10 = val_metrics.get("NDCG@10", 0.0)
            elapsed = time.time() - t0
            line = (f"Epoch {epoch:>4d}/{args.epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  " +
                    "  ".join(f"{k}: {v:.4f}" for k, v in sorted(val_metrics.items())))
            log(line, L)
            history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

            # Early stopping & checkpointing
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                patience_cnt = 0
                ckpt_path = os.path.join(args.save_dir, f"{tag}_best.pt")
                torch.save(model.state_dict(), ckpt_path)
                log(f"  [BEST] NDCG@10={best_ndcg:.4f}  saved -> {ckpt_path}", L)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    log(f"\n[train] Early stopping at epoch {epoch} (patience={args.patience})", L)
                    break

    # ── Final test evaluation ─────────────────────────────────────────────── #
    log("\n-- Final Test Evaluation --", L)
    ckpt_path = os.path.join(args.save_dir, f"{tag}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_dict, dataset.n_users, dataset.n_items, device, args.top_k)
    result_line = "[TEST] " + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(test_metrics.items()))
    log(result_line, L)

    # Save results
    results = {
        "model": tag,
        "config": vars(args),
        "test_metrics": test_metrics,
        "history": history,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n[train] Results saved -> results/{tag}_results.json", L)


if __name__ == "__main__":
    main()
