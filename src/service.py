"""Simple Python API used by the Streamlit demo (and usable from any script).

    from src.service import RecommenderService
    svc = RecommenderService()
    svc.recommend_for_user(user_id=1, n=10)
    svc.recommend_for_new_user(genres=["Sci-Fi"], age_group="25-34", gender="M",
                               ratings={"Star Wars (1977)": 5})

The demo service is trained on ALL MovieLens ratings (offline evaluation in
src/evaluate.py uses the train/test split instead). User ids are 1-based, as in the dataset.
"""
from __future__ import annotations

import pandas as pd

from .data import GENRES, load
from .models import Recommender, all_models


class RecommenderService:
    def __init__(self):
        self.ml = load()
        self.models: dict[str, Recommender] = all_models()
        for m in self.models.values():
            m.fit(self.ml.ratings, self.ml)
        self.default_model = "Hybrid (cold-start aware)"
        self.title_to_id = dict(zip(self.ml.items["title"], self.ml.items["item_id"]))
        self.genres = [g for g in GENRES if g != "unknown"]

    # ------------------------------------------------------------------ helpers
    def user_profile(self, user_id: int) -> dict[int, float]:
        g = self.ml.ratings[self.ml.ratings["user_id"] == user_id - 1]
        return dict(zip(g["item_id"], g["rating"]))

    def user_info(self, user_id: int) -> dict:
        row = self.ml.users.loc[user_id - 1]
        return {"age_group": row["age_group"], "gender": row["gender"],
                "age": int(row["age"]), "occupation": row["occupation"]}

    def _table(self, model: Recommender, profile, info, n) -> pd.DataFrame:
        recs = model.recommend(profile, n, info)
        items = self.ml.items
        return pd.DataFrame({
            "Rank": range(1, len(recs) + 1),
            "Movie": items.loc[recs, "title"].values,
            "Genres": [", ".join(items.at[i, "genre_list"]) for i in recs],
            "Avg rating": [round(model.mean_rating[i], 2) for i in recs],
            "# ratings": [int(model.counts[i]) for i in recs],
            "Why recommended": [model.explain(profile, i, info) for i in recs],
        })

    # ------------------------------------------------------------------ API
    def recommend_for_user(self, user_id: int, n: int = 10, model: str | None = None) -> pd.DataFrame:
        """Top-n recommendations for an existing MovieLens user (1-based id)."""
        m = self.models[model or self.default_model]
        return self._table(m, self.user_profile(user_id), self.user_info(user_id), n)

    def recommend_for_new_user(self, genres: list[str] | None = None, age_group: str | None = None,
                               gender: str | None = None, ratings: dict[str, float] | None = None,
                               n: int = 10, model: str | None = None) -> pd.DataFrame:
        """Cold start: a new user described only by sign-up info and (optionally) a few ratings."""
        m = self.models[model or self.default_model]
        profile = {self.title_to_id[t]: float(r) for t, r in (ratings or {}).items()}
        info = {"age_group": age_group, "gender": gender, "genres": genres or []}
        return self._table(m, profile, info, n)

    def top_rated_by_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        g = self.ml.ratings[self.ml.ratings["user_id"] == user_id - 1]
        g = g.sort_values(["rating", "timestamp"], ascending=[False, False]).head(n)
        return pd.DataFrame({"Movie": self.ml.items.loc[g["item_id"], "title"].values,
                             "Your rating": g["rating"].values})

    def popular_titles(self, n: int = 150) -> list[str]:
        counts = self.ml.ratings["item_id"].value_counts().head(n).index
        return sorted(self.ml.items.loc[counts, "title"])
