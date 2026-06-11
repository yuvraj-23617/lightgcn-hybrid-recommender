"""
semantic_encoder.py
===================
Sentence-BERT encoder for movie metadata.
Encodes each movie's (title + genres) into a fixed-size vector,
then projects to the model's embedding dimension.

Used to initialise E_item^(0) in LightGCN for the semantic extension.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


SBERT_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, accurate
CACHE_PATH  = "data/sbert_item_emb.pt"


def encode_items(
    dataset,
    emb_dim: int = 64,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Encode all items using Sentence-BERT and project to emb_dim.

    Parameters
    ----------
    dataset   : ML1MDataset  (needs .movies_df with title & genres)
    emb_dim   : target dimension to project into (must match LightGCN emb_dim)
    device    : torch device
    batch_size: encoding batch size
    use_cache : if True, save/load encoded vectors from CACHE_PATH

    Returns
    -------
    item_emb : torch.Tensor  shape (n_items, emb_dim)  on `device`
    """
    projected_cache = f"{CACHE_PATH.replace('.pt', '')}_dim{emb_dim}.pt"

    if use_cache and os.path.exists(projected_cache):
        print(f"[SBERT] Loading cached embeddings from {projected_cache}")
        return torch.load(projected_cache, map_location=device)

    # ── Build text corpus ──────────────────────────────────────────────────── #
    texts = []
    for item_idx in range(dataset.n_items):
        title  = dataset.get_movie_title(item_idx)
        genres = dataset.get_movie_genres(item_idx).replace("|", ", ")
        texts.append(f"{title}. Genres: {genres}")

    # ── Sentence-BERT encoding ─────────────────────────────────────────────── #
    print(f"[SBERT] Encoding {len(texts)} items with '{SBERT_MODEL}' …")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(SBERT_MODEL)

    raw_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="SBERT encoding"):
        batch = texts[i : i + batch_size]
        emb   = sbert.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        raw_embs.append(emb)

    raw_embs = np.vstack(raw_embs)                          # (n_items, 384)
    raw_tensor = torch.tensor(raw_embs, dtype=torch.float32)

    # ── Linear projection → emb_dim ───────────────────────────────────────── #
    sbert_dim = raw_tensor.shape[1]
    if sbert_dim != emb_dim:
        print(f"[SBERT] Projecting {sbert_dim}→{emb_dim} …")
        proj = nn.Linear(sbert_dim, emb_dim, bias=False)
        nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            projected = proj(raw_tensor)
    else:
        projected = raw_tensor

    # L2-normalise to keep scale compatible with random user embeddings
    projected = torch.nn.functional.normalize(projected, dim=1)

    # ── Cache ─────────────────────────────────────────────────────────────── #
    os.makedirs(os.path.dirname(projected_cache) or ".", exist_ok=True)
    torch.save(projected, projected_cache)
    print(f"[SBERT] Cached projected embeddings → {projected_cache}")

    return projected.to(device)
