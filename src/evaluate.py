"""Ranking metrics and offline evaluation (warm users + cold-start simulation).

Run:  python -m src.evaluate        -> writes reports/results_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import MovieLens, Split, load, temporal_split
from .models import Recommender, all_models

REPORTS = Path(__file__).resolve().parent.parent / "reports"
RELEVANT_RATING = 4  # a held-out movie counts as "relevant" if the user rated it >= 4


# --------------------------------------------------------------------------- metrics
def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(recs, relevant, k):
    return float(bool(set(recs[:k]) & relevant))


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def average_precision_at_k(recs, relevant, k):
    hits, total = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k) if relevant else 0.0


METRICS = {"Precision@{k}": precision_at_k, "Recall@{k}": recall_at_k,
           "NDCG@{k}": ndcg_at_k, "MAP@{k}": average_precision_at_k,
           "HitRate@{k}": hit_rate_at_k}


def evaluate_lists(rec_lists: list[list[int]], relevant_sets: list[set], k: int,
                   item_counts: np.ndarray, n_items: int) -> dict[str, float]:
    out = {name.format(k=k): float(np.mean([f(r, rel, k) for r, rel in zip(rec_lists, relevant_sets)]))
           for name, f in METRICS.items()}
    recommended = {i for r in rec_lists for i in r[:k]}
    out["Coverage"] = len(recommended) / n_items  # share of catalogue ever recommended
    # novelty: mean self-information -log2(p(item)); higher = less mainstream picks
    p = (item_counts + 1) / (item_counts.sum() + n_items)
    out["Novelty"] = float(np.mean([-np.log2(p[i]) for r in rec_lists for i in r[:k]]))
    return out


def user_info(ml: MovieLens, uid: int) -> dict:
    row = ml.users.loc[uid]
    return {"age_group": row["age_group"], "gender": row["gender"]}


# --------------------------------------------------------------------------- experiments
def evaluate_warm(models: dict[str, Recommender], split: Split, ml: MovieLens, k: int = 10) -> pd.DataFrame:
    """Existing users: profile = their training ratings, target = future ratings >= 4."""
    profiles = {u: dict(zip(g["item_id"], g["rating"])) for u, g in split.train.groupby("user_id")}
    rel = split.test[split.test["rating"] >= RELEVANT_RATING].groupby("user_id")["item_id"].apply(set)
    users = [u for u in rel.index if u in profiles]
    counts = np.bincount(split.train["item_id"], minlength=ml.n_items)
    rows = []
    for name, m in models.items():
        recs = [m.recommend(profiles[u], k, user_info(ml, u)) for u in users]
        rows.append({"Model": name, **evaluate_lists(recs, [rel[u] for u in users], k, counts, ml.n_items)})
    return pd.DataFrame(rows).set_index("Model")


def evaluate_cold(models: dict[str, Recommender], split: Split, ml: MovieLens,
                  known: tuple[int, ...] = (0, 1, 3, 5, 10, 20), k: int = 10,
                  which: str = "test") -> pd.DataFrame:
    """Simulated new users (never seen in training). We reveal only their first n ratings
    (n = 0 means we only know age group & gender from sign-up) and measure how well each
    model ranks the movies they went on to rate >= 4."""
    counts = np.bincount(split.train["item_id"], minlength=ml.n_items)
    rows = []
    for n in known:
        profiles, rels, uids = [], [], []
        for u, g in split.cold_subset(which).groupby("user_id"):
            later = g.iloc[n:]
            rel = set(later.loc[later["rating"] >= RELEVANT_RATING, "item_id"])
            if rel:
                profiles.append(dict(zip(g["item_id"].iloc[:n], g["rating"].iloc[:n])))
                rels.append(rel)
                uids.append(u)
        for name, m in models.items():
            recs = [m.recommend(p, k, user_info(ml, u)) for p, u in zip(profiles, uids)]
            res = evaluate_lists(recs, rels, k, counts, ml.n_items)
            rows.append({"Known ratings": n, "Model": name, **res})
    return pd.DataFrame(rows)


def run(k: int = 10, verbose: bool = True):
    ml = load()
    split = temporal_split(ml.ratings)
    models = all_models()
    for m in models.values():
        m.fit(split.train, ml)
    warm = evaluate_warm(models, split, ml, k)
    cold = evaluate_cold(models, split, ml, k=k)
    REPORTS.mkdir(exist_ok=True)
    warm.round(4).to_csv(REPORTS / "results_warm_users.csv")
    cold.round(4).to_csv(REPORTS / "results_cold_start.csv", index=False)
    if verbose:
        print(f"Train {len(split.train):,} | test {len(split.test):,} | "
              f"cold-start users {len(split.cold_users)} "
              f"({len(split.cold_val_users)} tuning / {len(split.cold_test_users)} test)\n")
        print("== Warm users ==")
        print(warm.round(4).to_string())
        print("\n== Cold start: NDCG@10 by number of known ratings ==")
        print(cold.pivot(index="Known ratings", columns="Model", values=f"NDCG@{k}").round(4).to_string())
    return warm, cold, models, split, ml


if __name__ == "__main__":
    run()
