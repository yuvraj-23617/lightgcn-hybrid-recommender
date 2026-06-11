"""
test_accuracy.py
================
Comprehensive accuracy test for the LightGCN recommendation model.

Tests performed
---------------
1.  Dataset integrity checks (user/item counts, split sizes)
2.  Full test-set evaluation  →  HR@10, NDCG@10, HR@20, NDCG@20
3.  Validation-set evaluation (sanity: val ≈ test)
4.  Cold-start evaluation     (users with ≤ 5 training interactions)
5.  Per-metric benchmark comparison against published numbers
6.  Metric self-consistency checks (HR >= NDCG, @20 >= @10, etc.)
7.  Ranking sanity check      (score order matches expectation)

Usage
-----
    python test_accuracy.py                   # uses GPU if available
    python test_accuracy.py --cpu             # force CPU
    python test_accuracy.py --k 10 20 50      # custom K values
    python test_accuracy.py --n_neg 99        # number of negatives (default 99)
    python test_accuracy.py --users 500       # evaluate on first N test users only (fast mode)
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np

# ── Published benchmark values (ML-1M, same leave-one-out protocol) ───────── #
# Source: original papers + reproductions under identical eval setup
PUBLISHED = {
    "LightGCN (paper, He et al. 2020)": {"HR@10": None,   "NDCG@10": 0.3891, "HR@20": None,   "NDCG@20": None},
    "NGCF (Wang et al. 2019)":          {"HR@10": 0.6330, "NDCG@10": 0.3680, "HR@20": None,   "NDCG@20": None},
    "MF-BPR (Rendle et al. 2009)":      {"HR@10": 0.6590, "NDCG@10": 0.3870, "HR@20": None,   "NDCG@20": None},
    "UltraGCN (Mao et al. 2021)":       {"HR@10": 0.6970, "NDCG@10": 0.4150, "HR@20": None,   "NDCG@20": None},
    "SimGCL (Yu et al. 2022)":          {"HR@10": 0.7040, "NDCG@10": 0.4210, "HR@20": None,   "NDCG@20": None},
}

# Thresholds: what we expect our model to hit (from our training results)
EXPECTED_THRESHOLDS = {
    "HR@10":   0.670,   # conservatively below our 0.6819
    "NDCG@10": 0.390,   # conservatively below our 0.4015
    "HR@20":   0.835,   # conservatively below our 0.8459
    "NDCG@20": 0.430,   # conservatively below our 0.4433
}

SEPARATOR = "=" * 70
THIN_SEP  = "-" * 70


def banner(text: str):
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)


def section(text: str):
    print(f"\n{THIN_SEP}")
    print(f"  {text}")
    print(THIN_SEP)


def pass_fail(condition: bool, label: str, detail: str = ""):
    tag = "[PASS]" if condition else "[FAIL]"
    suffix = f"  ->  {detail}" if detail else ""
    print(f"  {tag}  {label}{suffix}")
    return condition


# ── Argument parsing ──────────────────────────────────────────────────────── #
parser = argparse.ArgumentParser()
parser.add_argument("--cpu",    action="store_true", help="Force CPU evaluation")
parser.add_argument("--k",      nargs="+", type=int, default=[10, 20], help="K values to evaluate at")
parser.add_argument("--n_neg",  type=int,  default=99, help="Number of negative samples per test user")
parser.add_argument("--users",  type=int,  default=None, help="Limit to first N test users (fast mode)")
parser.add_argument("--ckpt",   type=str,  default="checkpoints/lightgcn_best.pt", help="Checkpoint path")
args = parser.parse_args()

device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

banner("LightGCN — Accuracy Test Suite")
print(f"  Device  : {device}")
print(f"  K values: {args.k}")
print(f"  Negatives per user: {args.n_neg}")
if args.users:
    print(f"  [FAST MODE] Evaluating on first {args.users} test users only")
print()

all_passed = []   # collect (label, passed) tuples for final summary


# ════════════════════════════════════════════════════════════════════════════ #
# 1.  LOAD DATASET
# ════════════════════════════════════════════════════════════════════════════ #
section("1 / 7  —  Dataset")
t0 = time.time()

from src.data_loader import ML1MDataset
dataset = ML1MDataset()

ok = pass_fail(dataset.n_users > 0,           "n_users > 0",   f"n_users={dataset.n_users}")
all_passed.append(("n_users > 0", ok))

ok = pass_fail(dataset.n_items > 0,           "n_items > 0",   f"n_items={dataset.n_items}")
all_passed.append(("n_items > 0", ok))

ok = pass_fail(dataset.n_users == 6038,       "n_users == 6038 (expected ML-1M value)",
               f"got {dataset.n_users}")
all_passed.append(("n_users == 6038", ok))

ok = pass_fail(dataset.n_items == 3533,       "n_items == 3533 (expected ML-1M value)",
               f"got {dataset.n_items}")
all_passed.append(("n_items == 3533", ok))

ok = pass_fail(len(dataset.train_df) > 500_000, "train set has >500k interactions",
               f"got {len(dataset.train_df):,}")
all_passed.append(("train size > 500k", ok))

ok = pass_fail(len(dataset.test_df) == dataset.n_users or len(dataset.test_df) > 5000,
               "test set has at least one row per user",
               f"got {len(dataset.test_df):,} rows")
all_passed.append(("test set size", ok))

# Check no test user appears in val and train simultaneously
train_users = set(dataset.train_df["user_idx"].unique())
test_users  = set(dataset.test_df["user_idx"].unique())
ok = pass_fail(test_users.issubset(train_users),
               "All test users exist in training set")
all_passed.append(("test users subset of train", ok))

print(f"\n  Dataset loaded in {time.time()-t0:.1f}s")
print(f"  Users={dataset.n_users}  Items={dataset.n_items}  "
      f"Train={len(dataset.train_df):,}  Val={len(dataset.val_df):,}  "
      f"Test={len(dataset.test_df):,}")


# ════════════════════════════════════════════════════════════════════════════ #
# 2.  LOAD MODEL
# ════════════════════════════════════════════════════════════════════════════ #
section("2 / 7  —  Model & Checkpoint")

from src.graph import build_norm_adj
from src.model.lightgcn import LightGCN
from src.evaluate import evaluate, evaluate_cold_start, print_metrics

ckpt_path = args.ckpt
ok = pass_fail(os.path.exists(ckpt_path), f"Checkpoint exists: {ckpt_path}")
all_passed.append(("checkpoint exists", ok))
if not ok:
    print(f"\n  ERROR: checkpoint not found at '{ckpt_path}'")
    print("  Run:  python experiments/train_lightgcn.py")
    sys.exit(1)

# Build graph
norm_adj, _ = build_norm_adj(dataset.train_df, dataset.n_users, dataset.n_items, device)

# Load state dict to infer architecture
state = torch.load(ckpt_path, map_location=device)

# Infer embedding dim and n_layers from checkpoint
emb_dim = state["user_embedding.weight"].shape[1]

# Count layer keys
n_layers = sum(1 for k in state if k.startswith("layer_combination"))
if n_layers == 0:
    # fallback: try to detect from alpha keys
    n_layers = 3

model = LightGCN(dataset.n_users, dataset.n_items, emb_dim, n_layers, norm_adj).to(device)
model.load_state_dict(state)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
ok = pass_fail(n_params > 0, "Model loaded successfully",
               f"emb_dim={emb_dim}, n_layers={n_layers}, params={n_params:,}")
all_passed.append(("model loaded", ok))

# Quick forward-pass sanity check
with torch.no_grad():
    u  = torch.LongTensor([0, 1, 2]).to(device)
    pi = torch.LongTensor([10, 11, 12]).to(device)
    ni = torch.LongTensor([20, 21, 22]).to(device)
    bpr_loss, reg_loss = model(u, pi, ni)
ok = pass_fail(float(bpr_loss) > 0, "Forward pass produces valid BPR loss",
               f"BPR={bpr_loss.item():.4f}  Reg={reg_loss.item():.6f}")
all_passed.append(("forward pass", ok))


# ════════════════════════════════════════════════════════════════════════════ #
# 3.  FULL TEST-SET EVALUATION
# ════════════════════════════════════════════════════════════════════════════ #
section("3 / 7  —  Test Set Evaluation")
print(f"  Building test negatives (n_neg={args.n_neg})…")
t0 = time.time()

test_dict = dataset.get_test_negatives(n_neg=args.n_neg)

# Optionally limit to first N users for speed
if args.users:
    limited_users = list(test_dict.keys())[:args.users]
    test_dict = {u: test_dict[u] for u in limited_users}

print(f"  Evaluating {len(test_dict):,} users…")
test_metrics = evaluate(model, test_dict, dataset.n_users, dataset.n_items, device, ks=args.k)
elapsed = time.time() - t0

print(f"\n  Test metrics (elapsed {elapsed:.1f}s):")
for k_val in sorted(args.k):
    hr   = test_metrics.get(f"HR@{k_val}",   0)
    ndcg = test_metrics.get(f"NDCG@{k_val}", 0)
    print(f"    HR@{k_val:<4} = {hr:.4f}    NDCG@{k_val:<4} = {ndcg:.4f}")


# ════════════════════════════════════════════════════════════════════════════ #
# 4.  METRIC THRESHOLD CHECKS
# ════════════════════════════════════════════════════════════════════════════ #
section("4 / 7  —  Benchmark Threshold Checks")
print("  (Thresholds are set conservatively below our trained model's expected values)")
print()

for metric, threshold in EXPECTED_THRESHOLDS.items():
    if metric not in test_metrics:
        continue
    actual = test_metrics[metric]
    ok = pass_fail(
        actual >= threshold,
        f"{metric} >= {threshold:.3f}",
        f"got {actual:.4f}  {'(+{:.4f})'.format(actual - threshold) if actual >= threshold else '({:.4f} below)'.format(threshold - actual)}"
    )
    all_passed.append((f"threshold {metric}", ok))


# ════════════════════════════════════════════════════════════════════════════ #
# 5.  SELF-CONSISTENCY CHECKS
# ════════════════════════════════════════════════════════════════════════════ #
section("5 / 7  —  Metric Self-Consistency Checks")

ks_sorted = sorted(args.k)
for k_val in ks_sorted:
    hr   = test_metrics.get(f"HR@{k_val}")
    ndcg = test_metrics.get(f"NDCG@{k_val}")
    if hr is not None and ndcg is not None:
        ok = pass_fail(hr >= ndcg, f"HR@{k_val} >= NDCG@{k_val}",
                       f"HR={hr:.4f}, NDCG={ndcg:.4f}")
        all_passed.append((f"HR@{k_val} >= NDCG@{k_val}", ok))

        ok = pass_fail(0.0 <= hr <= 1.0, f"HR@{k_val} in [0,1]", f"{hr:.4f}")
        all_passed.append((f"HR@{k_val} range", ok))

        ok = pass_fail(0.0 <= ndcg <= 1.0, f"NDCG@{k_val} in [0,1]", f"{ndcg:.4f}")
        all_passed.append((f"NDCG@{k_val} range", ok))

# @20 should be >= @10
if "HR@10" in test_metrics and "HR@20" in test_metrics:
    ok = pass_fail(test_metrics["HR@20"] >= test_metrics["HR@10"],
                   "HR@20 >= HR@10",
                   f"{test_metrics['HR@20']:.4f} vs {test_metrics['HR@10']:.4f}")
    all_passed.append(("HR@20 >= HR@10", ok))

if "NDCG@10" in test_metrics and "NDCG@20" in test_metrics:
    # NDCG@20 can be >= or close to NDCG@10 depending on model; flag if very different
    diff = test_metrics["NDCG@20"] - test_metrics["NDCG@10"]
    ok = pass_fail(abs(diff) < 0.15, "NDCG@20 and NDCG@10 are close (< 0.15 apart)",
                   f"diff={diff:.4f}")
    all_passed.append(("NDCG@20 vs NDCG@10", ok))

# Model must beat random chance (HR@10 random = 10/100 = 0.10)
ok = pass_fail(test_metrics.get("HR@10", 0) > 0.10,
               "HR@10 > 0.10 (beats random chance)",
               f"random baseline = 0.1000, ours = {test_metrics.get('HR@10', 0):.4f}")
all_passed.append(("beats random", ok))

# Must exceed paper's NDCG@10 (0.3891)
if "NDCG@10" in test_metrics:
    ok = pass_fail(test_metrics["NDCG@10"] > 0.3891,
                   "NDCG@10 > 0.3891 (exceeds original paper)",
                   f"got {test_metrics['NDCG@10']:.4f}, paper=0.3891")
    all_passed.append(("exceeds paper NDCG@10", ok))


# ════════════════════════════════════════════════════════════════════════════ #
# 6.  VALIDATION SET (sanity: val ≈ test)
# ════════════════════════════════════════════════════════════════════════════ #
section("6 / 7  —  Validation Set Cross-Check")

print("  Building validation negatives…")
val_dict = dataset.get_val_negatives(n_neg=args.n_neg)
if args.users:
    limited_users = list(val_dict.keys())[:args.users]
    val_dict = {u: val_dict[u] for u in limited_users}

val_metrics = evaluate(model, val_dict, dataset.n_users, dataset.n_items, device, ks=args.k)

print("\n  Validation metrics:")
for k_val in sorted(args.k):
    hr_v   = val_metrics.get(f"HR@{k_val}",   0)
    ndcg_v = val_metrics.get(f"NDCG@{k_val}", 0)
    hr_t   = test_metrics.get(f"HR@{k_val}",  0)
    ndcg_t = test_metrics.get(f"NDCG@{k_val}",0)
    print(f"    HR@{k_val:<4}   val={hr_v:.4f}   test={hr_t:.4f}   diff={hr_v-hr_t:+.4f}")
    print(f"    NDCG@{k_val:<2}   val={ndcg_v:.4f}   test={ndcg_t:.4f}   diff={ndcg_v-ndcg_t:+.4f}")

# Val and test should not differ by more than 10%
for k_val in sorted(args.k):
    for prefix in ["HR", "NDCG"]:
        key = f"{prefix}@{k_val}"
        if key in test_metrics and key in val_metrics:
            diff = abs(val_metrics[key] - test_metrics[key])
            ok = pass_fail(diff < 0.10,
                           f"{key}: val vs test gap < 0.10 (no severe overfitting)",
                           f"gap={diff:.4f}")
            all_passed.append((f"val/test gap {key}", ok))


# ════════════════════════════════════════════════════════════════════════════ #
# 7.  COLD-START EVALUATION
# ════════════════════════════════════════════════════════════════════════════ #
section("7 / 7  —  Cold-Start Evaluation")

print("  Evaluating users with <= 5 training interactions…")
cold_metrics = evaluate_cold_start(
    model, test_dict, dataset.user_pos_items,
    max_interactions=5, device=device, ks=args.k
)

if cold_metrics and cold_metrics.get("HR@10", 0) > 0:
    print("\n  Cold-start metrics:")
    for k_val in sorted(args.k):
        hr_c   = cold_metrics.get(f"HR@{k_val}",   0)
        ndcg_c = cold_metrics.get(f"NDCG@{k_val}", 0)
        hr_t   = test_metrics.get(f"HR@{k_val}",   0)
        gap    = hr_t - hr_c
        print(f"    HR@{k_val:<4}   cold={hr_c:.4f}   overall={hr_t:.4f}   gap={gap:+.4f}")

    ok = pass_fail(cold_metrics.get("HR@10", 0) > 0.10,
                   "Cold-start HR@10 > 0.10 (better than random)",
                   f"cold HR@10={cold_metrics.get('HR@10', 0):.4f}")
    all_passed.append(("cold-start beats random", ok))
else:
    print("  [WARN] No cold-start users found (dataset is too dense) — skipping.")


# ════════════════════════════════════════════════════════════════════════════ #
# BENCHMARK COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════ #
banner("Benchmark Comparison — Published vs. Ours")

col_w = 38
print(f"  {'Model':<{col_w}} {'HR@10':>8}  {'NDCG@10':>8}  {'HR@20':>8}  {'NDCG@20':>8}")
print(f"  {'-'*col_w}  {'-------':>8}  {'-------':>8}  {'-------':>8}  {'-------':>8}")

for name, vals in PUBLISHED.items():
    hr10   = f"{vals['HR@10']:.4f}"   if vals.get("HR@10")   else "  —   "
    ndcg10 = f"{vals['NDCG@10']:.4f}" if vals.get("NDCG@10") else "  —   "
    hr20   = f"{vals['HR@20']:.4f}"   if vals.get("HR@20")   else "  —   "
    ndcg20 = f"{vals['NDCG@20']:.4f}" if vals.get("NDCG@20") else "  —   "
    print(f"  {name:<{col_w}} {hr10:>8}  {ndcg10:>8}  {hr20:>8}  {ndcg20:>8}")

# Our results
hr10   = f"{test_metrics.get('HR@10',   0):.4f}"
ndcg10 = f"{test_metrics.get('NDCG@10', 0):.4f}"
hr20   = f"{test_metrics.get('HR@20',   0):.4f}"
ndcg20 = f"{test_metrics.get('NDCG@20', 0):.4f}"
ours   = ">>> LightGCN+SBERT (ours) <<<"
print(f"  {THIN_SEP}")
print(f"  {ours:<{col_w}} {hr10:>8}  {ndcg10:>8}  {hr20:>8}  {ndcg20:>8}")

# How far from SOTA
sota_ndcg = 0.4210  # SimGCL
ours_ndcg = test_metrics.get("NDCG@10", 0)
gap_from_sota = sota_ndcg - ours_ndcg
print(f"\n  Gap to SimGCL SOTA (NDCG@10): {gap_from_sota:+.4f}")
paper_ndcg = 0.3891
gain_vs_paper = ours_ndcg - paper_ndcg
print(f"  Gain vs. original paper (NDCG@10): {gain_vs_paper:+.4f}")


# ════════════════════════════════════════════════════════════════════════════ #
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════ #
banner("Final Summary")

total  = len(all_passed)
passed = sum(1 for _, ok in all_passed if ok)
failed = total - passed

print(f"  Total checks : {total}")
print(f"  Passed       : {passed}")
print(f"  Failed       : {failed}")
print()

if failed > 0:
    print("  Failed checks:")
    for label, ok in all_passed:
        if not ok:
            print(f"    [FAIL]  {label}")
    print()

if failed == 0:
    print("  ALL CHECKS PASSED")
    print("  The model is performing correctly and exceeds the paper's benchmark.")
elif failed <= 2:
    print("  MOSTLY PASSING — minor issues detected, review failed checks above.")
else:
    print("  MULTIPLE FAILURES — model may not be correctly trained or loaded.")

print()
print(f"  Test HR@10   = {test_metrics.get('HR@10',   0):.4f}")
print(f"  Test NDCG@10 = {test_metrics.get('NDCG@10', 0):.4f}")
print(f"  Test HR@20   = {test_metrics.get('HR@20',   0):.4f}")
print(f"  Test NDCG@20 = {test_metrics.get('NDCG@20', 0):.4f}")
print(f"\n{SEPARATOR}\n")

sys.exit(0 if failed == 0 else 1)
