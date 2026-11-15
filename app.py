"""Streamlit demo for the movie recommendation system.

Run:  streamlit run app.py
"""
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.service import RecommenderService

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
HYBRID = "Hybrid (cold-start aware)"

st.set_page_config(page_title="CineMatch · Movie Recommender", page_icon="🎬", layout="wide")

# fixed colour per model, identical in every chart (baseline in neutral grey)
MODEL_COLORS = {HYBRID: "#a78bfa", "SVD (matrix factorisation)": "#f97316", "Item-KNN CF": "#22c55e",
                "Content-based": "#eab308", "Popularity": "#8a8984"}

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
:root {
  --bg: #07070b; --card: #12121a; --card-2: #181824; --border: #26263a;
  --text: #ececf3; --muted: #9a9ab0; --violet: #8b5cf6; --violet-soft: #c4b5fd;
  --cyan: #22d3ee; --gold: #fbbf24; --pink: #f472b6;
}
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(139,92,246,.18), transparent 60%),
                      radial-gradient(900px 400px at -10% 10%, rgba(34,211,238,.10), transparent 60%), var(--bg); }
[data-testid="stSidebar"] { background: #0b0b12; border-right: 1px solid var(--border); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1400px; }

/* hero */
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(34,211,238,.10));
        border: 1px solid rgba(139,92,246,.35); }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em;
           background: linear-gradient(90deg, #fff, var(--violet-soft) 45%, var(--cyan));
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: .35rem 0 0; color: var(--muted); font-size: .98rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: var(--card); border: 1px solid var(--border); border-bottom: none;
                               border-radius: 12px 12px 0 0; padding: .55rem 1.1rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(180deg, rgba(139,92,246,.30), var(--card)); }

/* section titles */
.section { font-size: 1.25rem; font-weight: 700; margin: .4rem 0 .8rem; color: var(--text); }
.section small { color: var(--muted); font-weight: 400; font-size: .85rem; margin-left: .4rem; }

/* movie cards */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: .8rem; }
.card { display: flex; gap: .9rem; padding: .95rem 1rem; border-radius: 14px; background: var(--card);
        border: 1px solid var(--border); transition: transform .15s ease, border-color .15s ease; }
.card:hover { transform: translateY(-2px); border-color: rgba(139,92,246,.6); }
.rank { flex: 0 0 2.4rem; height: 2.4rem; border-radius: 10px; display: grid; place-items: center;
        font-weight: 800; color: #fff; background: linear-gradient(135deg, var(--violet), var(--cyan)); }
.body { min-width: 0; flex: 1; }
.title { font-weight: 700; font-size: 1.02rem; color: var(--text); line-height: 1.3; }
.meta { color: var(--muted); font-size: .82rem; margin: .25rem 0 .45rem; }
.meta b { color: var(--gold); font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: .3rem; margin-bottom: .5rem; }
.chip { font-size: .72rem; padding: .12rem .5rem; border-radius: 999px; color: var(--violet-soft);
        background: rgba(139,92,246,.14); border: 1px solid rgba(139,92,246,.3); }
.why { font-size: .84rem; color: #d6d6e4; line-height: 1.45; }
.why div { margin-top: .15rem; }

/* compact baseline list */
.mini { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: .5rem; }
.mini .item { padding: .55rem .75rem; border-radius: 10px; background: var(--card-2); border: 1px solid var(--border);
              font-size: .86rem; color: #c9c9d6; }
.mini .item span { color: var(--muted); margin-right: .35rem; font-weight: 700; }

/* profile panel */
.panel { padding: 1rem 1.1rem; border-radius: 14px; background: var(--card); border: 1px solid var(--border); }
.big { font-size: 2.3rem; font-weight: 800; line-height: 1; color: var(--text); }
.label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .25rem; }
.fav { display: flex; justify-content: space-between; padding: .35rem 0; border-bottom: 1px dashed var(--border);
       font-size: .86rem; color: #d6d6e4; }
.fav:last-child { border-bottom: none; }
.stars { color: var(--gold); white-space: nowrap; margin-left: .5rem; }

/* CF weight meter */
.meter { height: 10px; border-radius: 999px; background: #1f1f2e; overflow: hidden; margin: .35rem 0 .2rem; }
.meter > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--violet), var(--cyan)); }

/* KPI tiles */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .8rem; margin-bottom: 1rem; }
.kpi { padding: 1rem 1.1rem; border-radius: 14px; background: var(--card); border: 1px solid var(--border); }
.kpi .v { font-size: 1.9rem; font-weight: 800; color: var(--text); }
.kpi .d { font-size: .82rem; color: var(--muted); }
.kpi .up { color: #4ade80; font-weight: 700; }
.note { color: var(--muted); font-size: .88rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training models on MovieLens 100K ...")
def get_service() -> RecommenderService:
    return RecommenderService()


svc = get_service()
MODELS = list(svc.models)
REASON_ICONS = [("Because you liked", "🎯"), ("Matches your favourite genres", "🎭"),
                ("Popular", "🔥"), ("Fits your overall", "🧭"), ("Similar style", "🎞️")]


def reason_html(text: str) -> str:
    lines = []
    for part in [p for p in text.split(" · ") if p]:
        icon = next((i for prefix, i in REASON_ICONS if part.startswith(prefix)), "💡")
        lines.append(f"<div>{icon} {escape(part)}</div>")
    return "".join(lines) or "<div>💡 Recommended for you</div>"


def stars(r: float) -> str:
    full = int(round(r))
    return "★" * full + "☆" * (5 - full)


def cards(df: pd.DataFrame):
    html = ['<div class="grid">']
    for _, r in df.iterrows():
        chips = "".join(f'<span class="chip">{escape(g)}</span>' for g in r["Genres"].split(", ") if g)
        html.append(
            f'<div class="card"><div class="rank">{r["Rank"]}</div><div class="body">'
            f'<div class="title">{escape(r["Movie"])}</div>'
            f'<div class="meta"><b>★ {r["Avg rating"]:.2f}</b> &nbsp;·&nbsp; {r["# ratings"]} ratings</div>'
            f'<div class="chips">{chips}</div>'
            f'<div class="why">{reason_html(r["Why recommended"])}</div></div></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def mini_list(df: pd.DataFrame):
    items = "".join(f'<div class="item"><span>{r["Rank"]}</span>{escape(r["Movie"])}</div>' for _, r in df.iterrows())
    st.markdown(f'<div class="mini">{items}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f'<div class="section">{title}<small>{sub}</small></div>', unsafe_allow_html=True)


def short(name: str) -> str:
    return name.split(" (")[0]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("UneeQ Interns · ML Task: Recommendation System · MovieLens 100K")
    model_name = st.selectbox("Model", MODELS, index=MODELS.index(HYBRID),
                              help="The hybrid is the recommended model; the others are shown for comparison.")
    n = st.slider("Number of recommendations", 4, 20, 8, step=2)
    compare = st.toggle("Compare with popularity baseline", value=True)
    st.divider()
    st.markdown('<div class="note">Every recommendation explains itself:<br>'
                '🎯 similar to movies you liked<br>🎭 matches your genres<br>🔥 popular with people like you</div>',
                unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>Find your next favourite movie</h1>'
            '<p>Collaborative filtering + content-based + popularity, blended by how much we know about you. '
            'Works from your very first visit.</p></div>', unsafe_allow_html=True)

tab_existing, tab_new, tab_eval = st.tabs(["👤  Existing user", "✨  New user (cold start)", "📊  Evaluation"])

# ------------------------------------------------------------------ existing user
with tab_existing:
    left, right = st.columns([1, 3], gap="large")
    with left:
        user_id = st.number_input("User ID (1 – 943)", min_value=1, max_value=svc.ml.n_users, value=1)
        info = svc.user_info(user_id)
        profile = svc.user_profile(user_id)
        fav = svc.models["Content-based"].top_genres(profile, k=3)
        top = svc.top_rated_by_user(user_id, 6)
        favs = "".join(f'<div class="fav">{escape(m)}<span class="stars">{stars(r)}</span></div>'
                       for m, r in zip(top["Movie"], top["Your rating"]))
        chips = "".join(f'<span class="chip">{g}</span>' for g in fav)
        st.markdown(
            f'<div class="panel"><div class="label">Movies rated</div><div class="big">{len(profile)}</div>'
            f'<div class="meta">Age {info["age"]} · {"Female" if info["gender"] == "F" else "Male"} · '
            f'{info["occupation"]}</div><div class="label" style="margin-top:.6rem">Favourite genres</div>'
            f'<div class="chips">{chips or "—"}</div><div class="label" style="margin-top:.6rem">'
            f'Highest-rated</div>{favs}</div>', unsafe_allow_html=True)
    with right:
        section(f"Recommended for user {user_id}", f"· {model_name}")
        cards(svc.recommend_for_user(user_id, n, model_name))
        if compare and model_name != "Popularity":
            st.write("")
            section("Popularity baseline", "· the same list for everyone")
            mini_list(svc.recommend_for_user(user_id, n, "Popularity"))

# ------------------------------------------------------------------ new user
with tab_new:
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown('<div class="note">No rating history yet? Tell us a little about yourself. We start from what '
                    'people like you enjoy, then personalise with every movie you rate.</div>',
                    unsafe_allow_html=True)
        st.write("")
        age_group = st.selectbox("Age group", ["<18", "18-24", "25-34", "35-44", "45-54", "55+"], index=2)
        gender = st.radio("Gender", ["F", "M"], horizontal=True,
                          format_func=lambda g: "Female" if g == "F" else "Male")
        genres = st.multiselect("Favourite genres", svc.genres, default=["Sci-Fi", "Comedy"])
        rated = st.multiselect("Rate a few movies (optional)", svc.popular_titles(),
                               help="The more you rate, the more personal the recommendations get.")
        ratings = {t: st.slider(t, 1, 5, 4, key=f"r_{t}") for t in rated}
        a = svc.models[HYBRID].alpha(ratings)
        st.markdown(f'<div class="panel"><div class="label">Personalisation level</div>'
                    f'<div class="big">{a:.0%}</div><div class="meter"><div style="width:{a * 100:.0f}%"></div>'
                    f'</div><div class="note">Collaborative-filtering weight. It grows with every rating '
                    f'({len(ratings)} so far).</div></div>', unsafe_allow_html=True)
    with right:
        section("Recommended for you", f"· {model_name}")
        if model_name in ("Item-KNN CF", "SVD (matrix factorisation)") and not ratings:
            st.warning("Pure collaborative filtering can't recommend anything without ratings. This is the "
                       "cold-start problem. Rate some movies, or switch to the Hybrid model.", icon="🧊")
        elif model_name == "Content-based" and not (ratings or genres):
            st.warning("Pick at least one genre or rate a movie for content-based recommendations.", icon="🎭")
        else:
            cards(svc.recommend_for_new_user(genres, age_group, gender, ratings, n, model_name))
        if compare and model_name != "Popularity":
            st.write("")
            section("Popularity baseline", "· the same list for everyone")
            mini_list(svc.recommend_for_new_user(genres, age_group, gender, ratings, n, "Popularity"))

# ------------------------------------------------------------------ evaluation
with tab_eval:
    warm_csv, cold_csv = REPORTS / "results_warm_users.csv", REPORTS / "results_cold_start.csv"
    if not warm_csv.exists():
        st.info("Run `python -m src.evaluate` to generate the evaluation results.")
    else:
        warm = pd.read_csv(warm_csv)
        cold = pd.read_csv(cold_csv)
        w = warm.set_index("Model")
        lift = w.loc[HYBRID, "NDCG@10"] / w.loc["Popularity", "NDCG@10"] - 1
        cov = w.loc[HYBRID, "Coverage"] / w.loc["Popularity", "Coverage"]
        c3 = cold[cold["Known ratings"] == 3].set_index("Model")["NDCG@10"]
        st.markdown(f"""
        <div class="kpis">
          <div class="kpi"><div class="label">Hybrid NDCG@10</div><div class="v">{w.loc[HYBRID, 'NDCG@10']:.3f}</div>
            <div class="d"><span class="up">+{lift:.0%}</span> vs popularity baseline</div></div>
          <div class="kpi"><div class="label">Hit rate @10</div><div class="v">{w.loc[HYBRID, 'HitRate@10']:.0%}</div>
            <div class="d">of users get ≥ 1 movie they loved</div></div>
          <div class="kpi"><div class="label">Catalogue coverage</div><div class="v">{cov:.1f}×</div>
            <div class="d">more movies recommended than the baseline</div></div>
          <div class="kpi"><div class="label">New user, 3 ratings</div><div class="v">{c3[HYBRID]:.3f}</div>
            <div class="d">NDCG@10 vs {c3['Popularity']:.3f} for popularity</div></div>
        </div>""", unsafe_allow_html=True)

        order = ["Popularity", "Content-based", "Item-KNN CF", "SVD (matrix factorisation)", HYBRID]
        color = alt.Color("Model:N", scale=alt.Scale(domain=order, range=[MODEL_COLORS[m] for m in order]),
                          legend=None)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            section("Existing users", "· NDCG@10, higher is better")
            base = alt.Chart(warm).encode(
                y=alt.Y("Model:N", sort=order[::-1], title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("NDCG@10:Q", title=None, axis=alt.Axis(grid=True, gridOpacity=.15),
                        scale=alt.Scale(domain=[0, float(warm["NDCG@10"].max()) * 1.18])),
                tooltip=["Model", alt.Tooltip("NDCG@10:Q", format=".3f"),
                         alt.Tooltip("Recall@10:Q", format=".3f"), alt.Tooltip("HitRate@10:Q", format=".1%")])
            bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(color=color)
            labels = base.mark_text(align="left", dx=6, color="#ececf3").encode(text=alt.Text("NDCG@10:Q", format=".3f"))
            st.altair_chart((bars + labels).properties(height=260), width="stretch")
        with c2:
            section("New users", "· NDCG@10 vs. ratings known")
            cold_plot = cold[~((cold["Known ratings"] == 0) & (cold["NDCG@10"] == 0))]
            lines = alt.Chart(cold_plot).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
                x=alt.X("Known ratings:O", title="Ratings known about the new user", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("NDCG@10:Q", title=None, axis=alt.Axis(gridOpacity=.15, format=".2f", tickCount=5)),
                color=alt.Color("Model:N", scale=alt.Scale(domain=order, range=[MODEL_COLORS[m] for m in order]),
                                legend=alt.Legend(orient="right", title=None, labelLimit=220,
                                                  labelExpr="split(datum.label, ' (')[0]")),
                tooltip=["Model", "Known ratings", alt.Tooltip("NDCG@10:Q", format=".3f")])
            st.altair_chart(lines.properties(height=250), width="stretch")

        with st.expander("📋 Full metric tables"):
            st.dataframe(warm.set_index("Model").style.format("{:.3f}"), width="stretch")
            st.dataframe(cold.pivot(index="Known ratings", columns="Model", values="NDCG@10")[order]
                         .style.format("{:.3f}"), width="stretch")
        st.markdown('<div class="note">Time-based split: each user\'s newest 20% of ratings are hidden, and a '
                    'hidden movie rated ≥ 4★ counts as relevant. New users were never seen in training. '
                    'Full details: <code>reports/evaluation_report.md</code></div>', unsafe_allow_html=True)
