"""Tune the hybrid recommender on validation data only (never on the test sets).

Validation data:
  * warm validation: the most recent 10% of each existing user's *training* ratings
    (models are refit on the remaining 90% while tuning);
  * cold validation: the "tuning" half of the simulated new users.
The test ratings and the "test" half of the new users are never touched here.

Objective = 0.5 * NDCG@10 (warm validation) + 0.5 * mean NDCG@10 (cold validation,
averaged over 0/1/3/5/10/20 known ratings), so the hybrid must work for both groups.

Run:  python -m src.tune     -> writes reports/best_hybrid.json and reports/tuning_grid.csv
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .data import load, temporal_split
from .evaluate import REPORTS, RELEVANT_RATING, ndcg_at_k, user_info
from .models import (HYBRID_CONFIG, ContentBasedRecommender, DemographicPopularityRecommender,
                     HybridRecommender, ItemKNNRecommender, SVDRecommender, _minmax)

K = 10
KNOWN = (0, 1, 3, 5, 10, 20)
GRID = {
    "cf": ["knn", "svd"],
    "n0": [1, 3, 5, 10, 20, 50, 100],
    "content_weight": [0.0, 0.1, 0.3, 0.5],
    "pop_floor": [0.0, 0.1, 0.2, 0.3, 0.5],
}


def inner_split(train: pd.DataFrame, frac: float = 0.1):
    train = train.sort_values(["user_id", "timestamp"])
    rank = train.groupby("user_id").cumcount()
    size = train.groupby("user_id")["item_id"].transform("size")
    is_val = rank >= np.ceil(size * (1 - frac))
    return train[~is_val], train[is_val]


def build_cases(ml, split, fit_train, warm_val):
    """Pre-compute every component score once; blending is then cheap for each config."""
    comps = {"knn": ItemKNNRecommender().fit(fit_train, ml), "svd": SVDRecommender().fit(fit_train, ml),
             "content": ContentBasedRecommender().fit(fit_train, ml),
             "pop": DemographicPopularityRecommender().fit(fit_train, ml)}

    def case(profile, rel, uid, group):
        info = user_info(ml, uid)
        return {"group": group, "n": len(profile), "rel": rel, "seen": list(profile),
                **{name: _minmax(m.score(profile, info)) for name, m in comps.items()}}

    cases = []
    profiles = {u: dict(zip(g["item_id"], g["rating"])) for u, g in fit_train.groupby("user_id")}
    rel = warm_val[warm_val["rating"] >= RELEVANT_RATING].groupby("user_id")["item_id"].apply(set)
    for u, r in rel.items():
        if u in profiles:
            cases.append(case(profiles[u], r, u, "warm"))
    for n in KNOWN:
        for u, g in split.cold_subset("val").groupby("user_id"):
            later = g.iloc[n:]
            r = set(later.loc[later["rating"] >= RELEVANT_RATING, "item_id"])
            if r:
                cases.append(case(dict(zip(g["item_id"].iloc[:n], g["rating"].iloc[:n])), r, u, f"cold_{n}"))
    return cases


def ndcg_for(scores, c):
    s = scores.copy()
    if c["seen"]:
        s[c["seen"]] = -np.inf
    top = np.argpartition(-s, K)[:K]
    return ndcg_at_k(top[np.argsort(-s[top])].tolist(), c["rel"], K)


def objective(per_case: list[float], cases) -> dict:
    df = pd.DataFrame({"group": [c["group"] for c in cases], "ndcg": per_case})
    by = df.groupby("group")["ndcg"].mean()
    cold = by[[f"cold_{n}" for n in KNOWN]]
    return {"warm_ndcg": by["warm"], "cold_ndcg": cold.mean(),
            "score": 0.5 * by["warm"] + 0.5 * cold.mean(),
            **{f"cold_{n}": by[f"cold_{n}"] for n in KNOWN}}


def run():
    ml = load()
    split = temporal_split(ml.ratings)
    fit_train, warm_val = inner_split(split.train)
    cases = build_cases(ml, split, fit_train, warm_val)
    print(f"Tuning on {sum(c['group'] == 'warm' for c in cases)} warm-validation users and "
          f"{len(split.cold_val_users)} cold-validation users ({len(cases)} cases)")

    baseline = objective([ndcg_for(c["pop"], c) for c in cases], cases)
    rows = []
    for cf, n0, b, lam in itertools.product(*GRID.values()):
        per = []
        for c in cases:
            comp = {"cf": c[cf], "content": c["content"], "pop": c["pop"]}
            per.append(ndcg_for(HybridRecommender.blend(comp, c["n"] / (c["n"] + n0), b, lam), c))
        rows.append({"cf": cf, "n0": n0, "content_weight": b, "pop_floor": lam, **objective(per, cases)})

    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    REPORTS.mkdir(exist_ok=True)
    grid.round(4).to_csv(REPORTS / "tuning_grid.csv", index=False)
    best = grid.iloc[0]
    config = {"cf": best["cf"], "n0": float(best["n0"]), "content_weight": float(best["content_weight"]),
              "pop_floor": float(best["pop_floor"])}
    HYBRID_CONFIG.write_text(json.dumps(config, indent=2))

    show = ["cf", "n0", "content_weight", "pop_floor", "warm_ndcg", "cold_ndcg", "score"]
    print("\nTop 5 configurations (validation):")
    print(grid[show].head().round(4).to_string(index=False))
    print(f"\nDemographic-popularity reference on the same validation: "
          f"warm {baseline['warm_ndcg']:.4f} | cold {baseline['cold_ndcg']:.4f}")
    print(f"Saved best config -> {HYBRID_CONFIG.name}: {config}")
    return grid, config


if __name__ == "__main__":
    run()
