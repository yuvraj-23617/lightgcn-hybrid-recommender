"""
run_all.py
==========
Master script that runs the FULL pipeline end-to-end:
  1. Train LightGCN
  2. Train LightGCN + SBERT
  3. Train all baselines (MF, NCF, NGCF)
  4. Run ablation studies
  5. Run cold-start evaluation
  6. Print final comparison table

Usage
-----
    python run_all.py               # run everything
    python run_all.py --skip_train  # skip training, just print results
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import subprocess
import json
import argparse


def run(cmd: list[str], label: str):
    print(f"\n{'='*65}")
    print(f"  STEP: {label}")
    print(f"{'='*65}")
    result = subprocess.run(
        [sys.executable, "-X", "utf8"] + cmd,
        cwd=os.path.dirname(__file__) or ".",
    )
    if result.returncode != 0:
        print(f"\n[run_all] WARNING: '{label}' exited with code {result.returncode}")
    return result.returncode


def print_final_table():
    """Read all result JSON files and print a unified comparison table."""
    results = {}

    paths = {
        "MF":             "results/baseline_results.json",
        "NCF":            "results/baseline_results.json",
        "NGCF":           "results/baseline_results.json",
        "LightGCN":       "results/lightgcn_results.json",
        "LightGCN+SBERT": "results/lightgcn_sbert_results.json",
    }

    baseline_data = None
    if os.path.exists("results/baseline_results.json"):
        with open("results/baseline_results.json") as f:
            baseline_data = json.load(f)

    if baseline_data:
        for model_name in ["mf", "ncf", "ngcf"]:
            if model_name in baseline_data:
                results[model_name.upper()] = baseline_data[model_name]["test_metrics"]

    for tag, path in [("LightGCN", "results/lightgcn_results.json"),
                      ("LightGCN+SBERT", "results/lightgcn_sbert_results.json")]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results[tag] = data["test_metrics"]

    if not results:
        print("[run_all] No results found yet. Run training first.")
        return

    print("\n" + "=" * 72)
    print(f"  FINAL RESULTS — MovieLens-1M (Leave-One-Out, 99 negatives)")
    print("=" * 72)
    print(f"{'Model':<18} {'HR@10':>8} {'NDCG@10':>10} {'HR@20':>8} {'NDCG@20':>10}")
    print("-" * 72)
    for model, m in results.items():
        print(
            f"{model:<18} "
            f"{m.get('HR@10', 0):>8.4f} "
            f"{m.get('NDCG@10', 0):>10.4f} "
            f"{m.get('HR@20', 0):>8.4f} "
            f"{m.get('NDCG@20', 0):>10.4f}"
        )
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_train",     action="store_true", help="Skip training, just print results")
    p.add_argument("--skip_baselines", action="store_true", help="Skip baseline training")
    p.add_argument("--skip_ablations", action="store_true", help="Skip ablation studies")
    p.add_argument("--emb_dim",    type=int,   default=64)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=2048)
    args = p.parse_args()

    if not args.skip_train:
        # 1. LightGCN
        run([
            "experiments/train_lightgcn.py",
            "--emb_dim",    str(args.emb_dim),
            "--n_layers",   str(args.n_layers),
            "--epochs",     str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--eval_every", "5",
            "--patience",   "15",
        ], "Train LightGCN (standard)")

        # 2. LightGCN + SBERT
        run([
            "experiments/train_lightgcn.py",
            "--semantic",
            "--emb_dim",    str(args.emb_dim),
            "--n_layers",   str(args.n_layers),
            "--epochs",     str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--eval_every", "5",
            "--patience",   "15",
        ], "Train LightGCN + SBERT (semantic init)")

        if not args.skip_baselines:
            # 3. All baselines
            run([
                "experiments/train_baselines.py",
                "--model", "all",
                "--emb_dim",    str(args.emb_dim),
                "--epochs",     "100",
                "--batch_size", str(args.batch_size),
            ], "Train Baselines (MF, NCF, NGCF)")

    if not args.skip_ablations:
        # 4. Ablations
        run([
            "experiments/ablations.py",
            "--study", "all",
            "--epochs",     "80",
            "--batch_size", str(args.batch_size),
        ], "Ablation Studies (K and embedding dim)")

    # 5. Cold-start
    run(["experiments/cold_start.py"], "Cold-Start Evaluation")

    # 6. Final table
    print_final_table()

    print("\n[run_all] Done! Launch the demo with:")
    print("    streamlit run app/app.py")


if __name__ == "__main__":
    main()
