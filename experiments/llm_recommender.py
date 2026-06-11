"""
llm_recommender.py
==================
Evaluates the Groq LLM (llama-3.3-70b-versatile) as a RECOMMENDER MODEL
using the EXACT same evaluation pipeline as LightGCN:

  Protocol : leave-one-out, 99 randomly sampled negatives per user
  Metrics  : HR@10, NDCG@10, HR@20, NDCG@20

How it works
------------
For each test user the LLM receives:
  1. The user's watch history  (up to 15 movie titles + genres)
  2. 100 candidate movies      (1 positive + 99 negatives, shuffled)
  3. Task: rank the candidates from best to worst match

The LLM returns a JSON array of candidate IDs in ranked order.
We apply the same HR / NDCG functions from src/evaluate.py.

Results saved to: results/llm_results.json

Usage
-----
    python -X utf8 experiments/llm_recommender.py               # 200 users
    python -X utf8 experiments/llm_recommender.py --n_users 50  # quick test
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json, time, re, random, argparse
import numpy as np
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

from src.data_loader import ML1MDataset
from src.evaluate import hit_rate, ndcg

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL           = "llama-3.1-8b-instant"  # 500K tokens/day (vs 100K for 70b)
MAX_TOKENS           = 300          # top-20 list needs ~150 tokens — plenty of headroom
TEMPERATURE          = 0.0          # fully deterministic — reproducible
SLEEP_BETWEEN_CALLS  = 1.2          # seconds — rate-limit safety
MAX_HISTORY_MOVIES   = 10           # reduced to cut input tokens
TOP_K_RANK           = 20           # ask LLM for top-20 only (enough for HR@10/NDCG@10)
DEFAULT_N_USERS      = 200          # ~180K tokens total — within 500K/day limit


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_ranking_prompt(history_titles: list[str],
                         candidates: list[tuple],
                         top_k: int = TOP_K_RANK) -> str:
    """
    Build a token-efficient prompt asking the LLM for its top-K picks.

    Token budget per user (approx):
      - History  : 10 movies × ~12 tok = 120
      - Candidates: 100 items × ~8 tok = 800   (titles only, no genres)
      - Overhead  : ~120 tok
      - Output    : ~100 tok
      Total       ≈ 1,140 tokens/user → 200 users ≈ 228K tokens (within 500K/day)

    Parameters
    ----------
    history_titles : list of "Title (Genres)" strings  (genres here give taste context)
    candidates     : list of (local_id: int, title_string) tuples
    top_k          : how many top candidates to request
    """
    history_str    = "\n".join(f"  - {t}" for t in history_titles)
    # Candidates: titles only (strip genres to save ~30% tokens)
    candidates_str = "\n".join(
        f"  [{cid}] {title.split('(')[0].strip()}"   # keep title, drop genre
        for cid, title in candidates
    )

    return (
        "You are a movie recommender system.\n"
        "Given a user's watch history and a list of candidate movies, "
        f"return your top {top_k} picks for this user.\n\n"
        "USER WATCH HISTORY (movies they liked):\n"
        f"{history_str}\n\n"
        "CANDIDATE MOVIES:\n"
        f"{candidates_str}\n\n"
        f"TASK: Return ONLY a JSON array of the {top_k} best candidate IDs "
        "in order (best first).\n"
        f"The array must contain exactly {top_k} IDs from the list above.\n"
        "Example format: [12, 5, 37, 0, 88, ...]\n\n"
        "Return ONLY the JSON array — no explanation, no extra text."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_top_k(response_text: str,
                valid_ids: set[int]) -> list[int] | None:
    """
    Extract a top-K list of integer IDs from the LLM response.

    We now only ask for top-K (e.g. 20), not all 100, so:
      - Output tokens needed: ~150 instead of ~500
      - Truncation failures drop to near-zero
      - Still fully valid for HR@10 / NDCG@10 evaluation

    Strategy:
      1. Find the first [...] block
      2. Parse as JSON integers
      3. Keep only IDs that appear in valid_ids (deduplicated)
      4. Return None only if nothing at all was parseable
    """
    match = re.search(r'\[[\d,\s]+\]', response_text, re.DOTALL)
    if not match:
        return None
    try:
        raw = [int(x) for x in json.loads(match.group())]
    except (json.JSONDecodeError, ValueError):
        return None

    # Deduplicate while preserving order, filter to known IDs
    seen, valid = set(), []
    for x in raw:
        if x in valid_ids and x not in seen:
            valid.append(x)
            seen.add(x)

    return valid if valid else None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_llm(client: Groq,
                 dataset: ML1MDataset,
                 test_dict: dict,
                 n_users: int,
                 seed: int = 42) -> dict:
    """
    Evaluate LLM as a ranking model on `n_users` random test users.

    Returns
    -------
    dict : {"HR@10": float, "NDCG@10": float, "HR@20": float, "NDCG@20": float}
    """
    rng        = random.Random(seed)
    all_users  = list(test_dict.keys())
    eval_users = rng.sample(all_users, min(n_users, len(all_users)))

    results = {"HR@10": [], "NDCG@10": [], "HR@20": [], "NDCG@20": []}
    failed  = 0

    for u in tqdm(eval_users, desc="LLM Recommender eval"):
        pos_item, neg_items = test_dict[u]
        candidate_items     = [pos_item] + list(neg_items)   # 100 items

        # ── Watch history ────────────────────────────────────────────────── #
        history_idxs = list(dataset.user_pos_items.get(u, set()))
        rng.shuffle(history_idxs)
        history_idxs  = history_idxs[:MAX_HISTORY_MOVIES]
        history_titles = [
            f"{dataset.get_movie_title(i)} ({dataset.get_movie_genres(i)})"
            for i in history_idxs
        ] or ["(no history available)"]

        # ── Candidate list (local IDs 0-99, shuffled to avoid bias) ──────── #
        shuffled = list(enumerate(candidate_items))         # (local_id, item_idx)
        rng.shuffle(shuffled)
        local_to_item  = {lid: item for lid, item in shuffled}
        item_to_local  = {item: lid  for lid, item in shuffled}

        candidates = [
            (lid, f"{dataset.get_movie_title(item)} ({dataset.get_movie_genres(item)})")
            for lid, item in sorted(shuffled)               # present in sorted order
        ]

        prompt    = build_ranking_prompt(history_titles, candidates, top_k=TOP_K_RANK)
        valid_ids = set(range(len(candidate_items)))

        # ── LLM call ─────────────────────────────────────────────────────── #
        try:
            response     = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )
            raw_text     = response.choices[0].message.content.strip()
            top_k_local  = parse_top_k(raw_text, valid_ids)

            if top_k_local is None:
                failed += 1
                continue

            # Build full ranked list:
            #   LLM's top-K first, then remaining items in random order
            top_k_set    = set(top_k_local)
            remaining    = [lid for lid in valid_ids if lid not in top_k_set]
            random.shuffle(remaining)
            full_ranked  = top_k_local + remaining

            # Map local IDs → item indices → compute metrics
            ranked_items = [local_to_item[lid] for lid in full_ranked]

            for k in [10, 20]:
                top_k_items = ranked_items[:k]
                results[f"HR@{k}"].append(hit_rate(top_k_items, pos_item))
                results[f"NDCG@{k}"].append(ndcg(top_k_items, pos_item))

        except Exception as e:
            print(f"\n  [warn] User {u} failed: {e}", flush=True)
            failed += 1

        time.sleep(SLEEP_BETWEEN_CALLS)

    n_ok = len(eval_users) - failed
    print(f"\n[llm_eval] Evaluated {n_ok}/{len(eval_users)} users "
          f"({failed} failed / skipped)", flush=True)

    if not results["HR@10"]:
        return {"HR@10": 0.0, "NDCG@10": 0.0, "HR@20": 0.0, "NDCG@20": 0.0}

    return {k: float(np.mean(v)) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate LLM as a recommender model")
    p.add_argument("--n_users",  type=int, default=DEFAULT_N_USERS,
                   help=f"Users to evaluate (default {DEFAULT_N_USERS})")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--save_dir", default="results")
    args = p.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    client = Groq(api_key=api_key)
    print(f"[llm_recommender] Model    : {GROQ_MODEL}")
    print(f"[llm_recommender] Users    : {args.n_users}")
    print(f"[llm_recommender] Temp     : {TEMPERATURE}  (deterministic)", flush=True)

    print("[llm_recommender] Loading dataset ...", flush=True)
    dataset   = ML1MDataset(seed=args.seed)

    print("[llm_recommender] Building test negatives ...", flush=True)
    test_dict = dataset.get_test_negatives()

    print("[llm_recommender] Starting evaluation ...\n", flush=True)
    t0      = time.time()
    metrics = evaluate_llm(client, dataset, test_dict,
                           n_users=args.n_users, seed=args.seed)
    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print(f"  LLM Recommender Results  ({args.n_users} users, {elapsed/60:.1f} min)")
    print(f"{'='*55}")
    for k, v in sorted(metrics.items()):
        print(f"  {k:12s}: {v:.4f}")
    print(f"{'='*55}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    output = {
        "model"       : "LLM Recommender (llama-3.3-70b-versatile)",
        "config"      : {
            "llm_model"          : GROQ_MODEL,
            "n_users_evaluated"  : args.n_users,
            "temperature"        : TEMPERATURE,
            "candidates_per_user": 100,
            "note"               : "Same leave-one-out protocol as LightGCN"
        },
        "test_metrics": metrics,
    }
    out_path = os.path.join(args.save_dir, "llm_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[llm_recommender] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
