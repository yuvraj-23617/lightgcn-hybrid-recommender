"""
hybrid_recommender.py
=====================
Combines LightGCN embedding scores with LLM ranking scores:

    final_score = alpha * lightgcn_score_norm + (1 - alpha) * llm_score

Where:
  lightgcn_score_norm : dot-product of user/item embeddings, min-max normalised to [0,1]
  llm_score           : (n_candidates - rank) / n_candidates  (0-indexed rank from LLM)

If the LLM call fails for a user, the hybrid falls back to LightGCN scores only.

Usage
-----
    python -X utf8 experiments/hybrid_recommender.py
    python -X utf8 experiments/hybrid_recommender.py --alpha 0.7 --n_users 100
    python -X utf8 experiments/hybrid_recommender.py --alpha 0.5 --n_users 200
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json, time, random, argparse
import numpy as np
import torch
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

from src.data_loader import ML1MDataset
from src.graph import build_norm_adj
from src.model.lightgcn import LightGCN
from src.evaluate import hit_rate, ndcg
from experiments.llm_recommender import (
    build_ranking_prompt,
    parse_top_k,
    GROQ_MODEL,
    SLEEP_BETWEEN_CALLS,
    MAX_HISTORY_MOVIES,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K_RANK,
)

load_dotenv()

DEFAULT_ALPHA   = 0.7    # weight on LightGCN (0 = LLM only, 1 = LightGCN only)
DEFAULT_N_USERS = 200


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hybrid(
    client      : Groq,
    model       : LightGCN,
    dataset     : ML1MDataset,
    test_dict   : dict,
    n_users     : int,
    alpha       : float,
    seed        : int  = 42,
    device      : torch.device | None = None,
) -> dict:
    """
    Evaluate the LightGCN + LLM hybrid model.

    Returns
    -------
    dict : {"HR@10": float, "NDCG@10": float, "HR@20": float, "NDCG@20": float}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    user_emb, item_emb = model.get_all_embeddings()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    rng        = random.Random(seed)
    all_users  = list(test_dict.keys())
    eval_users = rng.sample(all_users, min(n_users, len(all_users)))

    results       = {"HR@10": [], "NDCG@10": [], "HR@20": [], "NDCG@20": []}
    llm_failures  = 0

    for u in tqdm(eval_users, desc=f"Hybrid (alpha={alpha:.2f})"):
        pos_item, neg_items = test_dict[u]
        candidate_items     = [pos_item] + list(neg_items)   # 100 items
        n_cands             = len(candidate_items)

        # ── LightGCN scores (dot product, normalised to [0,1]) ──────────── #
        u_emb      = user_emb[u].unsqueeze(0)                    # (1, d)
        i_emb      = item_emb[candidate_items]                   # (C, d)
        lgcn_raw   = (u_emb * i_emb).sum(dim=1).cpu().numpy()   # (C,)

        lo, hi     = lgcn_raw.min(), lgcn_raw.max()
        lgcn_norm  = (lgcn_raw - lo) / (hi - lo) if hi > lo else np.full(n_cands, 0.5)

        # ── Watch history ────────────────────────────────────────────────── #
        history_idxs  = list(dataset.user_pos_items.get(u, set()))
        rng.shuffle(history_idxs)
        history_titles = [
            f"{dataset.get_movie_title(i)} ({dataset.get_movie_genres(i)})"
            for i in history_idxs[:MAX_HISTORY_MOVIES]
        ] or ["(no history available)"]

        # ── Candidate list (shuffled to avoid position bias) ─────────────── #
        shuffled      = list(enumerate(candidate_items))
        rng.shuffle(shuffled)
        local_to_item = {lid: item for lid, item in shuffled}
        candidates    = [
            (lid, f"{dataset.get_movie_title(item)} ({dataset.get_movie_genres(item)})")
            for lid, item in sorted(shuffled)
        ]
        expected_ids  = set(range(n_cands))

        # ── LLM ranking ──────────────────────────────────────────────────── #
        prompt   = build_ranking_prompt(history_titles, candidates, top_k=TOP_K_RANK)
        valid_ids = set(range(n_cands))

        ranked_local = None
        try:
            prompt   = build_ranking_prompt(history_titles, candidates, top_k=TOP_K_RANK)
            response = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )
            raw_text     = response.choices[0].message.content.strip()
            top_k_local  = parse_top_k(raw_text, valid_ids)
            if top_k_local is not None:
                # expand to full ranking: LLM top-K first, rest random
                top_k_set    = set(top_k_local)
                remaining    = [lid for lid in valid_ids if lid not in top_k_set]
                random.shuffle(remaining)
                ranked_local = top_k_local + remaining
        except Exception as e:
            print(f"\n  [warn] LLM failed for user {u}: {e}", flush=True)

        if ranked_local is None:
            # Fallback: use LightGCN ranking only
            llm_failures += 1
            order        = np.argsort(-lgcn_norm)
            ranked_items = [candidate_items[i] for i in order]
        else:
            # Convert LLM rank → score: best rank gets highest score
            llm_scores = np.zeros(n_cands)
            for rank, lid in enumerate(ranked_local):
                llm_scores[lid] = (n_cands - rank) / n_cands     # 1.0 → 0.0

            # Combine
            final_scores = alpha * lgcn_norm + (1.0 - alpha) * llm_scores
            order        = np.argsort(-final_scores)
            ranked_items = [candidate_items[i] for i in order]

        # ── Metrics ──────────────────────────────────────────────────────── #
        for k in [10, 20]:
            top_k = ranked_items[:k]
            results[f"HR@{k}"].append(hit_rate(top_k, pos_item))
            results[f"NDCG@{k}"].append(ndcg(top_k, pos_item))

        time.sleep(SLEEP_BETWEEN_CALLS)

    n_ok = len(eval_users) - llm_failures
    print(
        f"\n[hybrid] {n_ok}/{len(eval_users)} users had LLM ranking  "
        f"({llm_failures} fell back to LightGCN only)", flush=True
    )

    return {k: float(np.mean(v)) for k, v in results.items() if v}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="LightGCN + LLM Hybrid Recommender")
    p.add_argument("--alpha",    type=float, default=DEFAULT_ALPHA,
                   help="LightGCN weight 0-1 (default 0.7)")
    p.add_argument("--n_users",  type=int,   default=DEFAULT_N_USERS)
    p.add_argument("--ckpt",     default="checkpoints/lightgcn_best.pt")
    p.add_argument("--emb_dim",  type=int,   default=64)
    p.add_argument("--n_layers", type=int,   default=3)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--save_dir", default="results")
    args = p.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    client = Groq(api_key=api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hybrid] Device    : {device}")
    print(f"[hybrid] Alpha     : {args.alpha}  (LightGCN weight)")
    print(f"[hybrid] LLM model : {GROQ_MODEL}", flush=True)

    print("[hybrid] Loading dataset ...", flush=True)
    dataset   = ML1MDataset(seed=args.seed)
    test_dict = dataset.get_test_negatives()

    print("[hybrid] Building graph ...", flush=True)
    norm_adj, _ = build_norm_adj(
        dataset.train_df, dataset.n_users, dataset.n_items, device
    )

    print(f"[hybrid] Loading checkpoint: {args.ckpt} ...", flush=True)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = LightGCN(
        dataset.n_users, dataset.n_items,
        args.emb_dim, args.n_layers, norm_adj
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    print(f"[hybrid] Starting evaluation ({args.n_users} users) ...\n", flush=True)
    t0      = time.time()
    metrics = evaluate_hybrid(
        client, model, dataset, test_dict,
        n_users=args.n_users,
        alpha=args.alpha,
        seed=args.seed,
        device=device,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print(f"  Hybrid Results  (alpha={args.alpha}, {args.n_users} users, {elapsed/60:.1f} min)")
    print(f"{'='*55}")
    for k, v in sorted(metrics.items()):
        print(f"  {k:12s}: {v:.4f}")
    print(f"{'='*55}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    output = {
        "model"       : f"Hybrid LightGCN+LLM (alpha={args.alpha})",
        "config"      : {
            "alpha"             : args.alpha,
            "n_users_evaluated" : args.n_users,
            "llm_model"         : GROQ_MODEL,
            "lightgcn_ckpt"     : args.ckpt,
            "note"              : "final_score = alpha*lgcn_norm + (1-alpha)*llm_score"
        },
        "test_metrics": metrics,
    }
    out_path = os.path.join(args.save_dir, "hybrid_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[hybrid] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
