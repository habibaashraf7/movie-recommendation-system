"""Recommendation models.

Every model works on a *profile*: a dict {item_id: rating} of what a user has rated.
Because scoring only needs the profile (not a trained user id), the same code serves
existing users, brand-new users who rated a few movies, and users with no ratings.

    model.fit(train_df, movielens)
    scores = model.score(profile, user_info)        # np.ndarray, one score per item
    model.explain(profile, item_id, user_info)      # human-readable reason
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .data import GENRES, MovieLens

Profile = dict  # {item_id: rating}


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class Recommender:
    name = "base"

    def fit(self, train: pd.DataFrame, ml: MovieLens) -> "Recommender":
        self.ml = ml
        self.n_items = ml.n_items
        self.titles = ml.items["title"].to_numpy()
        counts = np.bincount(train["item_id"], minlength=self.n_items)
        sums = np.bincount(train["item_id"], weights=train["rating"], minlength=self.n_items)
        self.counts = counts
        self.mean_rating = np.divide(sums, counts, out=np.zeros(self.n_items), where=counts > 0)
        return self

    def score(self, profile: Profile, user_info: dict | None = None) -> np.ndarray:
        raise NotImplementedError

    def explain(self, profile: Profile, item: int, user_info: dict | None = None) -> str:
        return ""

    def recommend(self, profile: Profile, n: int = 10, user_info: dict | None = None,
                  exclude_seen: bool = True) -> list[int]:
        s = self.score(profile, user_info).astype(float).copy()
        if exclude_seen and profile:
            s[list(profile)] = -np.inf
        top = np.argpartition(-s, n)[:n]
        return top[np.argsort(-s[top])].tolist()


# --------------------------------------------------------------------------- baselines
class PopularityRecommender(Recommender):
    """Baseline: rank movies by how many people rated them (same list for everyone)."""
    name = "Popularity"

    def score(self, profile, user_info=None):
        return self.counts.astype(float)

    def explain(self, profile, item, user_info=None):
        return (f"Popular: rated by {self.counts[item]} users "
                f"(avg {self.mean_rating[item]:.1f}★)")


class DemographicPopularityRecommender(Recommender):
    """Cold-start helper: most popular movies among users of the same age group & gender."""
    name = "Demographic popularity"

    def fit(self, train, ml):
        super().fit(train, ml)
        df = train.merge(ml.users[["user_id", "age_group", "gender"]], on="user_id")
        self.group_counts = {
            key: np.bincount(g["item_id"], minlength=self.n_items).astype(float)
            for key, g in df.groupby(["age_group", "gender"], observed=True)
        }
        return self

    def _key(self, user_info):
        if not user_info:
            return None
        key = (user_info.get("age_group"), user_info.get("gender"))
        return key if key in self.group_counts else None

    def score(self, profile, user_info=None):
        key = self._key(user_info)
        base = self.counts.astype(float)
        if key is None:
            return base
        # blend in a little global popularity so small groups don't produce noisy lists
        return _minmax(self.group_counts[key]) + 0.2 * _minmax(base)

    def explain(self, profile, item, user_info=None):
        key = self._key(user_info)
        if key is None:
            return PopularityRecommender.explain(self, profile, item)
        n = int(self.group_counts[key][item])
        return f"Popular with {key[1]}, age {key[0]} viewers ({n} of them rated it)"


# --------------------------------------------------------------------------- content-based
class ContentBasedRecommender(Recommender):
    """Recommend movies whose genres / era match the movies the user liked.

    Item features = IDF-weighted genre flags + release-decade one-hot (L2-normalised).
    User profile  = sum of item features weighted by (rating - 3): liked movies pull
    the profile towards their genres, disliked movies push it away.
    """
    name = "Content-based"

    def __init__(self, decade_weight: float = 0.5):
        self.decade_weight = decade_weight

    def fit(self, train, ml):
        super().fit(train, ml)
        g = ml.items[GENRES].to_numpy(float)
        idf = np.log(len(g) / (1 + g.sum(0)))
        g = g * idf
        decade = (ml.items["year"].fillna(1990) // 10 * 10).astype(int)
        self.decades = sorted(decade.unique())
        d = pd.get_dummies(decade).reindex(columns=self.decades, fill_value=0).to_numpy(float)
        feats = np.hstack([g, self.decade_weight * d])
        self.features = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)
        self.genre_cols = len(GENRES)
        return self

    def user_vector(self, profile: Profile, genres: list[str] | None = None) -> np.ndarray:
        v = np.zeros(self.features.shape[1])
        for item, r in profile.items():
            v += (r - 3.0) * self.features[item]
        if genres:  # explicit preferences from a new user's onboarding form
            for gname in genres:
                v[GENRES.index(gname)] += 1.0
        return v

    def score(self, profile, user_info=None):
        genres = (user_info or {}).get("genres")
        v = self.user_vector(profile, genres)
        if not v.any():
            return np.zeros(self.n_items)
        return self.features @ v

    def top_genres(self, profile, user_info=None, k=3) -> list[str]:
        v = self.user_vector(profile, (user_info or {}).get("genres"))[: self.genre_cols]
        order = np.argsort(-v)
        return [GENRES[i] for i in order[:k] if v[i] > 0]

    def explain(self, profile, item, user_info=None):
        liked = set(self.top_genres(profile, user_info, k=5))
        match = [g for g in self.ml.items.at[item, "genre_list"] if g in liked]
        if match:
            return "Matches your favourite genres: " + ", ".join(match)
        return "Similar style to movies you rated"


# --------------------------------------------------------------------------- collaborative
class ItemKNNRecommender(Recommender):
    """Item-item collaborative filtering ("people who liked X also liked Y").

    Similarity = cosine between item columns of the user x item interaction matrix,
    keeping only the k nearest neighbours of each item. A user's score for item i is
    sum_j w_j * sim(j, i) over the movies j in their profile, with w_j = rating_j - 2.5
    (so 4-5 star ratings count positively and 1-2 star ratings count negatively).
    """
    name = "Item-KNN CF"

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, train, ml):
        super().fit(train, ml)
        X = sparse.csr_matrix((np.ones(len(train)), (train["user_id"], train["item_id"])),
                              shape=(ml.n_users, ml.n_items))
        co = (X.T @ X).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        if self.k and self.k < ml.n_items:  # keep top-k neighbours per item
            cut = np.argpartition(-sim, self.k, axis=0)[self.k:]
            np.put_along_axis(sim, cut, 0.0, axis=0)
        self.sim = sim
        return self

    @staticmethod
    def _weights(profile):
        items = np.fromiter(profile.keys(), int)
        w = np.fromiter(profile.values(), float) - 2.5
        return items, w

    def score(self, profile, user_info=None):
        if not profile:
            return np.zeros(self.n_items)
        items, w = self._weights(profile)
        return w @ self.sim[items]

    def explain(self, profile, item, user_info=None):
        if not profile:
            return ""
        items, w = self._weights(profile)
        contrib = w * self.sim[items, item]
        top = [items[i] for i in np.argsort(-contrib)[:2] if contrib[i] > 0]
        if not top:
            return ""
        return "Because you liked " + " and ".join(f"“{self.titles[j]}”" for j in top)


class SVDRecommender(Recommender):
    """Matrix factorisation (PureSVD). Users and movies are mapped to f latent 'taste'
    factors; any profile can be folded in as x @ V^T @ V without retraining."""
    name = "SVD (matrix factorisation)"

    def __init__(self, factors: int = 50, seed: int = 42):
        self.factors, self.seed = factors, seed

    def fit(self, train, ml):
        super().fit(train, ml)
        from sklearn.decomposition import TruncatedSVD
        X = sparse.csr_matrix((train["rating"].astype(float), (train["user_id"], train["item_id"])),
                              shape=(ml.n_users, ml.n_items))
        svd = TruncatedSVD(n_components=self.factors, random_state=self.seed).fit(X)
        self.V = svd.components_  # factors x items
        self._knn = ItemKNNRecommender().fit(train, ml)  # only used for explanations
        return self

    def score(self, profile, user_info=None):
        if not profile:
            return np.zeros(self.n_items)
        x = np.zeros(self.n_items)
        x[list(profile)] = list(profile.values())
        return (x @ self.V.T) @ self.V

    def explain(self, profile, item, user_info=None):
        if not profile:
            return ""
        return self._knn.explain(profile, item) or "Fits your overall taste profile"


# --------------------------------------------------------------------------- hybrid
class HybridRecommender(Recommender):
    """Cold-start-aware hybrid that shifts weight as we learn more about the user.

    personal = a * CF + (1 - a) * (b * content + (1 - b) * popularity),   a = n / (n + n0)
    score    = (1 - lam) * personal + lam * popularity

    n = number of ratings in the profile. With n = 0 the model is pure (demographic)
    popularity + genre preferences; as ratings accumulate, collaborative filtering
    takes over. `lam` keeps a small, constant popularity prior for everyone.
    All component scores are min-max normalised before blending.
    n0, b, lam and the CF model are tuned on validation users (see src/tune.py).
    """
    name = "Hybrid (cold-start aware)"

    def __init__(self, cf: Recommender | None = None, n0: float = 5.0,
                 content_weight: float = 0.3, pop_floor: float = 0.0):
        self.cf = cf or ItemKNNRecommender()
        self.n0, self.b, self.lam = n0, content_weight, pop_floor

    def fit(self, train, ml):
        super().fit(train, ml)
        self.cf.fit(train, ml)
        self.content = ContentBasedRecommender().fit(train, ml)
        self.pop = DemographicPopularityRecommender().fit(train, ml)
        return self

    def alpha(self, profile) -> float:
        n = len(profile)
        return n / (n + self.n0)

    def components(self, profile, user_info=None) -> dict[str, np.ndarray]:
        return {"cf": _minmax(self.cf.score(profile, user_info)),
                "content": _minmax(self.content.score(profile, user_info)),
                "pop": _minmax(self.pop.score(profile, user_info))}

    @staticmethod
    def blend(c: dict, a: float, b: float, lam: float) -> np.ndarray:
        b = b if c["content"].any() else 0.0
        personal = a * c["cf"] + (1 - a) * (b * c["content"] + (1 - b) * c["pop"])
        return (1 - lam) * personal + lam * c["pop"]

    def score(self, profile, user_info=None):
        return self.blend(self.components(profile, user_info), self.alpha(profile), self.b, self.lam)

    def explain(self, profile, item, user_info=None):
        reasons = [self.cf.explain(profile, item, user_info),
                   self.content.explain(profile, item, user_info) if (profile or (user_info or {}).get("genres")) else "",
                   self.pop.explain(profile, item, user_info) if self.alpha(profile) < 0.5 else ""]
        return " · ".join(r for r in reasons if r and not r.startswith("Similar style"))


HYBRID_CONFIG = Path(__file__).resolve().parent.parent / "reports" / "best_hybrid.json"


def make_hybrid(config: dict | None = None) -> HybridRecommender:
    """Build the hybrid from a config dict, or from the tuned config saved by src/tune.py."""
    if config is None:
        config = json.loads(HYBRID_CONFIG.read_text()) if HYBRID_CONFIG.exists() else {}
    cf = SVDRecommender() if config.get("cf") == "svd" else ItemKNNRecommender()
    return HybridRecommender(cf=cf, n0=config.get("n0", 5.0),
                             content_weight=config.get("content_weight", 0.3),
                             pop_floor=config.get("pop_floor", 0.0))


def all_models() -> dict[str, Recommender]:
    return {m.name: m for m in [PopularityRecommender(), ContentBasedRecommender(),
                                ItemKNNRecommender(), SVDRecommender(), make_hybrid()]}
