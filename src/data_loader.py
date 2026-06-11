"""
data_loader.py
==============
Downloads MovieLens-1M, converts to implicit feedback, performs leave-one-out
split, and provides DataLoader utilities for training.

Implicit feedback rule: rating >= 4  →  positive interaction
Evaluation protocol: leave-one-out with 99 randomly sampled negatives per test user
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m")


# ---------------------------------------------------------------------------
# Download & raw parsing
# ---------------------------------------------------------------------------

def download_ml1m(data_dir: str = DATA_DIR) -> str:
    """Download and unzip MovieLens-1M if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    ratings_path = os.path.join(data_dir, "ratings.dat")
    if os.path.exists(ratings_path):
        print("[data_loader] Found existing ratings.dat — skipping download.")
        return data_dir

    zip_path = os.path.join(data_dir, "ml-1m.zip")
    print(f"[data_loader] Downloading MovieLens-1M from {ML1M_URL} …")
    urllib.request.urlretrieve(ML1M_URL, zip_path)

    print("[data_loader] Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.join(data_dir, ".."))
    os.rename(os.path.join(data_dir, "..", "ml-1m"), data_dir) if not os.path.exists(data_dir) else None
    os.remove(zip_path)
    print("[data_loader] Done.")
    return data_dir


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load raw ratings.dat → DataFrame with columns [user_id, item_id, rating, timestamp]."""
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load movies.dat → DataFrame with columns [item_id, title, genres]."""
    path = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


def load_users(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load users.dat → DataFrame with columns [user_id, gender, age, occupation, zip]."""
    path = os.path.join(data_dir, "users.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_implicit_feedback(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit feedback.
    Keep only interactions where rating >= threshold.
    Re-index user_id and item_id to start from 0.
    """
    df = df[df["rating"] >= threshold].copy()
    df = df.sort_values("timestamp")

    # Re-index
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}

    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)

    return df, user2idx, item2idx


def leave_one_out_split(df: pd.DataFrame):
    """
    Leave-one-out split.
    For each user, the last interaction (by timestamp) goes to test,
    the second-to-last goes to validation, the rest to train.

    Returns:
        train_df, val_df, test_df  (all with user_idx, item_idx columns)
    """
    df = df.sort_values(["user_idx", "timestamp"])

    train_rows, val_rows, test_rows = [], [], []
    for _, group in df.groupby("user_idx"):
        rows = group.to_dict("records")
        if len(rows) < 3:
            train_rows.extend(rows)
            continue
        test_rows.append(rows[-1])
        val_rows.append(rows[-2])
        train_rows.extend(rows[:-2])

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    test_df  = pd.DataFrame(test_rows)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Dataset class (PyTorch-style)
# ---------------------------------------------------------------------------

class ML1MDataset:
    """
    Holds the full processed dataset and provides sampling utilities.

    Attributes
    ----------
    n_users, n_items : int
    train_df, val_df, test_df : pd.DataFrame
    user_pos_items : dict[int, set[int]]   (all positive items per user in train)
    """

    def __init__(self, data_dir: str = DATA_DIR, threshold: int = 4, seed: int = 42):
        np.random.seed(seed)
        download_ml1m(data_dir)

        ratings_df = load_ratings(data_dir)
        implicit_df, self.user2idx, self.item2idx = build_implicit_feedback(ratings_df, threshold)
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.idx2item = {v: k for k, v in self.item2idx.items()}

        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)

        self.train_df, self.val_df, self.test_df = leave_one_out_split(implicit_df)

        # Build positive item sets per user (train only)
        self.user_pos_items: dict[int, set] = defaultdict(set)
        for row in self.train_df.itertuples():
            self.user_pos_items[row.user_idx].add(row.item_idx)

        # Also include val & test items in "all positives" for negative sampling
        self.user_all_pos: dict[int, set] = defaultdict(set)
        for row in self.train_df.itertuples():
            self.user_all_pos[row.user_idx].add(row.item_idx)
        for row in self.val_df.itertuples():
            self.user_all_pos[row.user_idx].add(row.item_idx)
        for row in self.test_df.itertuples():
            self.user_all_pos[row.user_idx].add(row.item_idx)

        # Load movie metadata
        movies_df = load_movies(data_dir)
        movies_df["item_idx"] = movies_df["item_id"].map(self.item2idx)
        movies_df = movies_df.dropna(subset=["item_idx"])
        movies_df["item_idx"] = movies_df["item_idx"].astype(int)
        self.movies_df = movies_df.set_index("item_idx")

        print(
            f"[ML1MDataset] Users: {self.n_users}, Items: {self.n_items}, "
            f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}"
        )

    def get_movie_title(self, item_idx: int) -> str:
        try:
            return self.movies_df.loc[item_idx, "title"]
        except KeyError:
            return f"Movie {item_idx}"

    def get_movie_genres(self, item_idx: int) -> str:
        try:
            return self.movies_df.loc[item_idx, "genres"]
        except KeyError:
            return "Unknown"

    def sample_bpr_batch(self, batch_size: int):
        """
        Vectorised BPR batch sampler — much faster than a Python loop.
        Returns numpy arrays of shape (batch_size,).
        """
        all_users = np.array(list(self.user_pos_items.keys()))

        # Sample users with replacement
        user_indices = np.random.choice(len(all_users), size=batch_size, replace=True)
        users = all_users[user_indices]

        pos_items = np.zeros(batch_size, dtype=np.int64)
        neg_items = np.zeros(batch_size, dtype=np.int64)

        for i, u in enumerate(users):
            pos_set = self.user_pos_items[u]
            pos_items[i] = np.random.choice(list(pos_set))

            # Rejection sampling for negative (fast for sparse graphs)
            neg_j = np.random.randint(0, self.n_items)
            while neg_j in self.user_all_pos[u]:
                neg_j = np.random.randint(0, self.n_items)
            neg_items[i] = neg_j

        return users, pos_items, neg_items


    def get_test_negatives(self, n_neg: int = 99, seed: int = 42) -> dict:
        """
        For each user in test set, sample n_neg negative items.
        Returns dict: { user_idx: (pos_item_idx, [neg_item_idx, ...]) }
        """
        rng = np.random.default_rng(seed)
        test_dict = {}
        for row in self.test_df.itertuples():
            u = row.user_idx
            pos = row.item_idx
            neg_pool = list(set(range(self.n_items)) - self.user_all_pos[u])
            neg_samples = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False).tolist()
            test_dict[u] = (pos, neg_samples)
        return test_dict

    def get_val_negatives(self, n_neg: int = 99, seed: int = 123) -> dict:
        """Same as get_test_negatives but for validation set."""
        rng = np.random.default_rng(seed)
        val_dict = {}
        for row in self.val_df.itertuples():
            u = row.user_idx
            pos = row.item_idx
            neg_pool = list(set(range(self.n_items)) - self.user_all_pos[u])
            neg_samples = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False).tolist()
            val_dict[u] = (pos, neg_samples)
        return val_dict
