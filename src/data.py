"""Download, load and split the MovieLens 100K dataset."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ML_DIR = DATA_DIR / "ml-100k"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download(force: bool = False) -> Path:
    """Download and extract MovieLens 100K into ./data (skipped if already present)."""
    if ML_DIR.exists() and not force:
        return ML_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_DIR)
    return ML_DIR


@dataclass
class MovieLens:
    ratings: pd.DataFrame  # user_id, item_id, rating, timestamp (0-based ids)
    items: pd.DataFrame    # item_id, title, year, genres..., genre_list
    users: pd.DataFrame    # user_id, age, gender, occupation, age_group

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)


def load() -> MovieLens:
    download()
    ratings = pd.read_csv(ML_DIR / "u.data", sep="\t",
                          names=["user_id", "item_id", "rating", "timestamp"])
    items = pd.read_csv(ML_DIR / "u.item", sep="|", encoding="latin-1", header=None,
                        names=["item_id", "title", "release_date", "video_release", "url"] + GENRES)
    users = pd.read_csv(ML_DIR / "u.user", sep="|",
                        names=["user_id", "age", "gender", "occupation", "zip"])

    # switch to 0-based contiguous ids so they can index numpy arrays directly
    for df in (ratings, items, users):
        for col in ("user_id", "item_id"):
            if col in df:
                df[col] = df[col] - 1

    items["year"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce").dt.year
    items["genre_list"] = items[GENRES].apply(lambda r: [g for g in GENRES if r[g] == 1], axis=1)
    items = items.drop(columns=["video_release", "url"])
    users["age_group"] = pd.cut(users["age"], bins=[0, 18, 25, 35, 45, 55, 100],
                                labels=["<18", "18-24", "25-34", "35-44", "45-54", "55+"])
    return MovieLens(ratings, items, users)


@dataclass
class Split:
    train: pd.DataFrame        # all training interactions
    test: pd.DataFrame         # held-out (later) interactions of warm users
    cold_users: np.ndarray     # users fully removed from training (simulated new users)
    cold_ratings: pd.DataFrame  # all ratings of the cold users, time-ordered
    cold_val_users: np.ndarray  # half of the cold users: used ONLY for tuning
    cold_test_users: np.ndarray  # other half: used ONLY for the final report

    def cold_subset(self, which: str) -> pd.DataFrame:
        users = self.cold_val_users if which == "val" else self.cold_test_users
        return self.cold_ratings[self.cold_ratings["user_id"].isin(users)]


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2,
                   cold_frac: float = 0.2, seed: int = 42) -> Split:
    """Per-user temporal split + a group of users held out entirely for cold-start tests.

    * Warm users: the earliest (1 - test_frac) of each user's ratings go to train,
      the most recent test_frac go to test (we predict the *future*, not random gaps).
    * Cold users: a random cold_frac of users are removed from training completely.
    """
    rng = np.random.default_rng(seed)
    users = ratings["user_id"].unique()
    cold_users = np.sort(rng.choice(users, size=int(len(users) * cold_frac), replace=False))

    is_cold = ratings["user_id"].isin(cold_users)
    warm = ratings[~is_cold].sort_values(["user_id", "timestamp"])
    rank = warm.groupby("user_id").cumcount()
    size = warm.groupby("user_id")["item_id"].transform("size")
    in_test = rank >= np.ceil(size * (1 - test_frac))

    cold = ratings[is_cold].sort_values(["user_id", "timestamp"])
    shuffled = rng.permutation(cold_users)
    half = len(shuffled) // 2
    return Split(train=warm[~in_test].reset_index(drop=True),
                 test=warm[in_test].reset_index(drop=True),
                 cold_users=cold_users,
                 cold_ratings=cold.reset_index(drop=True),
                 cold_val_users=np.sort(shuffled[:half]),
                 cold_test_users=np.sort(shuffled[half:]))


if __name__ == "__main__":
    ml = load()
    print(f"{len(ml.ratings):,} ratings | {ml.n_users} users | {ml.n_items} movies")
