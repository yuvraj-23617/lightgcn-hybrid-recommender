"""
consolidate_results.py
======================
Reads all result JSON files and writes a single markdown table
suitable for the final report / demo slides.

Usage
-----
    python consolidate_results.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = "results"


def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main():
    rows = []

    # Baselines
    baselines = load("baseline_results.json")
    if baselines:
        for name in ["mf", "ncf", "ngcf"]:
            if name in baselines:
                m = baselines[name].get("test_metrics", {})
                rows.append({"Model": name.upper(), **m})

    # LightGCN
    lgcn = load("lightgcn_results.json")
    if lgcn:
        rows.append({"Model": "LightGCN (ours)", **lgcn.get("test_metrics", {})})

    # LightGCN + SBERT
    lgcn_s = load("lightgcn_sbert_results.json")
    if lgcn_s:
        rows.append({"Model": "LightGCN+SBERT (ours)", **lgcn_s.get("test_metrics", {})})

    # LLM Recommender
    llm_r = load("llm_results.json")
    if llm_r:
        rows.append({"Model": "LLM Recommender (Groq)", **llm_r.get("test_metrics", {})})

    # Hybrid (LightGCN + LLM)
    hyb_r = load("hybrid_results.json")
    if hyb_r:
        cfg   = hyb_r.get("config", {})
        alpha = cfg.get("alpha", "?")
        rows.append({"Model": f"Hybrid LightGCN+LLM (α={alpha})", **hyb_r.get("test_metrics", {})})

    if not rows:
        print("No results found. Train the models first.")
        return

    metrics = ["HR@10", "NDCG@10", "HR@20", "NDCG@20"]

    # Find best per metric
    best = {}
    for metric in metrics:
        best[metric] = max(r.get(metric, 0) for r in rows)

    # Markdown table
    header = "| Model | " + " | ".join(metrics) + " |"
    sep    = "|-------|" + "|".join(["------:"] * len(metrics)) + "|"
    lines  = [header, sep]

    for row in rows:
        cells = []
        for m in metrics:
            val = row.get(m, 0)
            s = fmt(val)
            if val == best[m]:
                s = f"**{s}**"   # bold best value
            cells.append(s)
        lines.append(f"| {row['Model']} | " + " | ".join(cells) + " |")

    md_table = "\n".join(lines)
    print("\n=== RESULTS TABLE (Markdown) ===\n")
    print(md_table)

    # Save to file
    out_path = os.path.join(RESULTS_DIR, "final_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# LightGCN — Final Results\n\n")
        f.write("**Dataset:** MovieLens-1M  \n")
        f.write("**Protocol:** Leave-one-out, 99 sampled negatives  \n\n")
        f.write(md_table)
        f.write("\n\n> Bold values = best per metric\n")
    print(f"\nSaved -> {out_path}")

    # Ablations
    abl = load("ablation_results.json")
    if abl:
        print("\n=== ABLATION: K (Layers) ===")
        if "layers" in abl:
            print(f"{'K':>4}  {'NDCG@10':>10}  {'HR@10':>8}")
            for k in sorted(abl["layers"].keys(), key=int):
                m = abl["layers"][k]
                print(f"{k:>4}  {m.get('NDCG@10', 0):>10.4f}  {m.get('HR@10', 0):>8.4f}")

        print("\n=== ABLATION: Embedding Dim ===")
        if "dim" in abl:
            print(f"{'d':>6}  {'NDCG@10':>10}  {'HR@10':>8}")
            for d in sorted(abl["dim"].keys(), key=int):
                m = abl["dim"][d]
                print(f"{d:>6}  {m.get('NDCG@10', 0):>10.4f}  {m.get('HR@10', 0):>8.4f}")


if __name__ == "__main__":
    main()
