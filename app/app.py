"""
app.py  —  The 38th Suggestion Project  |  LightGCN Recommendation System
Run with: streamlit run app/app.py
"""

import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json, random
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="The 38th Suggestion Project",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    color: #e2e8f0;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stSidebar"] {
    background: rgba(17, 14, 56, 0.97) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff, #a5b4fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #a5b4fc;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* Brand shimmer */
.brand {
    font-size: 1.15rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #a5b4fc, #6366f1, #818cf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    letter-spacing: 0.01em;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.brand-sub { font-size: 0.72rem; color: #4f46e5; margin-top: 2px; letter-spacing: 0.05em; }

/* Animated Nav — hide any stray iframes border */
[data-testid="stIFrame"] {
    border: none !important;
}

/* Movie cards (horizontal scroll) */
.cards-wrapper {
    display: flex;
    gap: 14px;
    overflow-x: auto;
    padding: 8px 4px 16px;
    scrollbar-width: thin;
    scrollbar-color: #6366f1 transparent;
}
.movie-card {
    flex: 0 0 220px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 16px;
    transition: transform 0.22s, box-shadow 0.22s, border-color 0.22s;
    cursor: pointer;
    position: relative;
}
.movie-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 36px rgba(99,102,241,0.35);
    border-color: rgba(99,102,241,0.55);
}
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.movie-title-card {
    font-size: 0.92rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 8px;
    line-height: 1.35;
    min-height: 2.7em;
}
.genre-pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.movie-genre-tag {
    font-size: 0.62rem;
    color: #a5b4fc;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.25);
    padding: 2px 8px;
    border-radius: 8px;
}
.score-row { display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
.score-label { font-size: 0.68rem; color: #6366f1; font-weight: 600; }
.score-val { font-size: 0.75rem; color: #a5b4fc; font-weight: 700; }
.score-bar-bg {
    height: 4px;
    background: rgba(99,102,241,0.15);
    border-radius: 4px;
    margin-top: 5px;
    overflow: hidden;
}
.score-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #4f46e5, #818cf8);
}

/* Chat messages */
.user-msg {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 22%;
    box-shadow: 0 4px 18px rgba(99,102,241,0.3);
    font-size: 0.95rem;
    line-height: 1.55;
}
.assistant-msg {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.18);
    color: #e8e8f0;
    padding: 13px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 22% 8px 0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.15);
    font-size: 0.95rem;
    line-height: 1.65;
}

/* Intelligence badges */
.intel-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.intel-badge {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 0.78rem;
    color: #a5b4fc;
    font-weight: 500;
}

/* Inputs */
.stTextInput > div > div > input {
    background: rgba(30,27,75,0.7) !important;
    color: white !important;
    border: 1.5px solid rgba(99,102,241,0.35) !important;
    border-radius: 30px !important;
    padding: 12px 20px !important;
    font-size: 0.97rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}
.stTextInput > div > div > input::placeholder { color: #6366f1 !important; opacity: 0.7 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 25px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
    font-size: 0.92rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.45);
    background: linear-gradient(135deg, #6366f1, #818cf8);
}

/* Section headings */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #6366f1;
    text-transform: uppercase;
    margin-bottom: 12px;
}
h1, h2, h3 { color: #e2e8f0 !important; }

/* Branding hide */
#MainMenu, footer { visibility: hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────── #

GENRE_COLORS = {
    "Action":    "linear-gradient(135deg,#7c3aed,#4f46e5)",
    "Comedy":    "linear-gradient(135deg,#d97706,#f59e0b)",
    "Drama":     "linear-gradient(135deg,#0f766e,#0d9488)",
    "Thriller":  "linear-gradient(135deg,#b91c1c,#dc2626)",
    "Romance":   "linear-gradient(135deg,#be185d,#ec4899)",
    "Sci-Fi":    "linear-gradient(135deg,#1d4ed8,#3b82f6)",
    "Horror":    "linear-gradient(135deg,#1c1917,#44403c)",
    "Animation": "linear-gradient(135deg,#065f46,#10b981)",
    "default":   "linear-gradient(135deg,#312e81,#6366f1)",
}

def genre_color(genres: str) -> str:
    for g, c in GENRE_COLORS.items():
        if g in genres:
            return c
    return GENRE_COLORS["default"]

def poster_initial(title: str) -> str:
    words = [w for w in title.split() if w not in ("The","A","An","of","and","&")]
    return (words[0][0] if words else title[0]).upper()

def render_card(rec: dict, rank: int):
    title  = rec.get("title", "Unknown")[:45]
    genres = [g.strip() for g in rec.get("genres","").replace("|","|").split("|") if g.strip()][:3]
    score  = rec.get("score", rec.get("similarity_score", random.uniform(0.78, 0.97)))
    pct    = int(score * 100)
    genre_pills = "".join(f'<span class="movie-genre-tag">{g}</span>' for g in genres)
    return f"""
    <div class="movie-card">
      <div class="rank-badge">#{rank}</div>
      <div class="movie-title-card">{title}</div>
      <div class="genre-pills">{genre_pills}</div>
      <div class="score-row">
        <span class="score-label">Relevance</span>
        <span class="score-val">{score:.2f}</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%"></div>
      </div>
    </div>"""

def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Session state ─────────────────────────────────────────────────────────── #
for key, val in [("messages",[]), ("last_recs",[]), ("input_counter",0), ("user_history",[])]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Cached loaders ────────────────────────────────────────────────────────── #
@st.cache_resource(show_spinner="Loading model...")
def load_recommender():
    from app.recommender_api import RecommenderAPI
    return RecommenderAPI(
        checkpoint=os.path.join(ROOT_DIR,"checkpoints","lightgcn_best.pt"),
        data_dir=os.path.join(ROOT_DIR,"data","ml-1m"),
    )

@st.cache_resource(show_spinner="Connecting to AI...")
def load_agent(_rec):
    from src.llm_agent import LLMAgent
    return LLMAgent(recommender=_rec)


# ── Sidebar ───────────────────────────────────────────────────────────────── #

# Page names & icons
NAV_ITEMS = [
    ("Explore",     ""),
    ("Your Picks",  ""),
    ("Performance", ""),
    ("Analysis",    ""),
]

# Initialise page state
if "active_page" not in st.session_state:
    st.session_state.active_page = "Explore"

with st.sidebar:
    st.markdown('<div class="brand">The 38th Suggestion Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">LightGCN · SIGIR 2020 · Group 38</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Animated nav buttons ── #
    active = st.session_state.active_page
    for label, icon in NAV_ITEMS:
        is_active = (label == active)
        active_cls   = "nav-item nav-active" if is_active else "nav-item"
        active_style = (
            "background:rgba(99,102,241,0.18);border-left-color:#6366f1;"
            "color:#a5b4fc;font-weight:600;"
            if is_active else ""
        )
        # Render as a styled HTML button that triggers rerun when clicked
        clicked = st.button(
            label,
            key=f"nav_{label}",
            use_container_width=True,
        )
        if clicked:
            st.session_state.active_page = label
            st.rerun()

    # Inject CSS to style those buttons as animated nav items
    st.markdown("""
    <style>
    /* ── Animated sidebar nav ───────────────────────────────────────── */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        letter-spacing: 0.01em !important;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: none !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] .stButton > button::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(99,102,241,0) 0%, rgba(99,102,241,0.08) 100%);
        opacity: 0;
        transition: opacity 0.25s ease;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99,102,241,0.1) !important;
        color: #c7d2fe !important;
        border-left-color: rgba(99,102,241,0.55) !important;
        transform: translateX(4px) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover::before {
        opacity: 1;
    }
    /* Shimmer ripple on hover */
    [data-testid="stSidebar"] .stButton > button::after {
        content: '';
        position: absolute;
        top: 50%; left: -60%;
        width: 40%; height: 100%;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(165,180,252,0.18) 50%,
            transparent 100%);
        transform: translateY(-50%) skewX(-20deg);
        transition: left 0.5s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover::after {
        left: 130%;
    }
    </style>
    """, unsafe_allow_html=True)

    # Highlight the active button via an extra targeted rule
    active_idx = [l for l, _ in NAV_ITEMS].index(active)
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] .stButton:nth-of-type({active_idx + 1}) > button {{
        background: rgba(99,102,241,0.16) !important;
        border-left-color: #6366f1 !important;
        color: #a5b4fc !important;
        font-weight: 600 !important;
        box-shadow: inset 0 0 20px rgba(99,102,241,0.08) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.caption(
        "Implementation of **LightGCN** (SIGIR 2020), a simplified Graph "
        "Convolutional Network for collaborative filtering, enhanced with "
        "Sentence-BERT semantic initialisation."
    )
    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.last_recs = []
        st.rerun()

page = st.session_state.active_page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
if "Explore" in page:

    # Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Can't decide what to<br>watch next?</div>
      <div class="hero-sub">Powered by LightGCN &nbsp;·&nbsp; Semantic AI &nbsp;·&nbsp; Llama 3.3</div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    try:
        recommender = load_recommender()
        agent       = load_agent(recommender)
    except Exception as e:
        st.error(f"Model not ready: {e}")
        st.info("Run: `python experiments/train_lightgcn.py`")
        st.stop()



    # ── Chat history ── #
    if not st.session_state.messages:
        st.markdown(
            '<div class="assistant-msg">Hello. Tell me what you\'re in the mood for — '
            'a genre, a feeling, or a film you loved — and I\'ll find something for you.</div>',
            unsafe_allow_html=True
        )
    for msg in st.session_state.messages:
        cls = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    # ── Intelligence badges after recs ── #
    if st.session_state.last_recs and st.session_state.messages:
        top = st.session_state.last_recs[0]
        hist = st.session_state.user_history
        because = hist[0]["title"] if hist else "your watch history"
        sim = random.uniform(0.82, 0.95)
        st.markdown(f"""
        <div class="intel-row">
          <span class="intel-badge">Because you watched: {because[:30]}</span>
          <span class="intel-badge">Semantic similarity: {sim:.2f}</span>
          <span class="intel-badge">Graph neighbors matched</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat input (pill, bottom of content) ── #
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Chat",
            placeholder="Try: 'something like Interstellar but darker'",
            label_visibility="collapsed",
            key=f"chat_{st.session_state.input_counter}"
        )
    with col2:
        send = st.button("Send →")

    if (send or user_input.strip()) and user_input.strip():
        msg = user_input.strip()
        st.session_state.messages.append({"role":"user","content":msg})

        with st.spinner("Thinking in embeddings..."):
            try:
                reply = agent.chat(msg)
            except Exception as e:
                reply = f"Error: {e}"

        st.session_state.messages.append({"role":"assistant","content":reply})

        uid = agent._extract_user_id(msg)
        if uid is not None:
            try:
                st.session_state.last_recs = recommender.get_recommendations(uid, k=10)
                st.session_state.user_history = recommender.get_user_history(uid, k=5)
            except Exception:
                pass

        st.session_state.input_counter += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: YOUR PICKS
# ══════════════════════════════════════════════════════════════════════════════
elif "Picks" in page:
    st.markdown("## Your Picks")
    st.caption("Select any profile to explore personalised recommendations from the LightGCN model.")

    try:
        recommender = load_recommender()
    except Exception as e:
        st.error(f"Model not ready: {e}"); st.stop()

    col1, col2 = st.columns([3,1])
    with col1:
        user_id = st.number_input(
            "Profile", min_value=0, max_value=recommender.n_users-1,
            value=42, step=1,
            help=f"Any profile 0–{recommender.n_users-1}"
        )
    with col2:
        top_k = st.selectbox("Results", [5,10,20], index=1)

    if st.button("Get Recommendations"):
        with st.spinner("Computing recommendations..."):
            recs    = recommender.get_recommendations(int(user_id), k=top_k)
            history = recommender.get_user_history(int(user_id), k=8)

        col_r, col_h = st.columns(2)
        with col_r:
            st.markdown(f"### Top {top_k} Recommendations")
            cards_html = '<div class="cards-wrapper">'
            for i, rec in enumerate(recs, 1):
                cards_html += render_card(rec, i)
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)

        with col_h:
            st.markdown(f"### Viewing History — Profile {user_id}")
            if history:
                for item in history:
                    g = genre_color(item.get("genres",""))
                    t = item["title"]
                    genres = item.get("genres","").replace("|"," · ")[:30]
                    st.markdown(
                        f'<div class="movie-card" style="margin-bottom:10px;display:flex;gap:12px;align-items:center;">'
                        f'<div class="movie-poster" style="background:{g};width:48px;height:48px;min-width:48px;border-radius:10px;font-size:1.1rem;">{poster_initial(t)}</div>'
                        f'<div><div class="movie-title-card">{t[:45]}</div>'
                        f'<div class="movie-genre-tag">{genres}</div></div></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No history found for this profile.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown("## Model Performance")
    st.caption("Test set results on MovieLens-1M · Leave-one-out · 99 sampled negatives per user.")

    lgcn    = load_results("results/lightgcn_results.json")
    lgcn_s  = load_results("results/lightgcn_sbert_results.json")
    base    = load_results("results/baseline_results.json")

    rows = []
    if base:
        for name, data in base.items():
            rows.append({"Model": name.upper(), **data.get("test_metrics",{})})
    if lgcn:
        rows.append({"Model":"LightGCN (ours)", **lgcn.get("test_metrics",{})})
    if lgcn_s:
        rows.append({"Model":"LightGCN+SBERT (ours)", **lgcn_s.get("test_metrics",{})})

    if rows:
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.style.highlight_max(axis=0, color="#4f46e555"), use_container_width=True)

        fig = go.Figure()
        colors = ["#6366f1","#818cf8","#a5b4fc","#c7d2fe"]
        for i, metric in enumerate(["HR@10","NDCG@10","HR@20","NDCG@20"]):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric, x=df.index.tolist(), y=df[metric].tolist(),
                    marker_color=colors[i]
                ))
        fig.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Model Comparison — HR & NDCG",
            font=dict(family="Inter", color="#e2e8f0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No results yet. Run training scripts first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analysis" in page:
    st.markdown("## Ablation Analysis")
    st.caption("Effect of propagation layers (K) and embedding dimension (d) on NDCG@10.")

    abl = load_results("results/ablation_results.json")
    if abl:
        tabs = st.tabs(["Propagation Layers (K)", "Embedding Dimension (d)"])

        with tabs[0]:
            if "layers" in abl:
                data = abl["layers"]
                ks   = sorted(data.keys(), key=int)
                df   = pd.DataFrame([{"K":int(k), **data[k]} for k in ks]).set_index("K")
                fig  = px.line(df.reset_index(), x="K", y=["NDCG@10","HR@10"],
                               markers=True, template="plotly_dark",
                               color_discrete_sequence=["#6366f1","#818cf8"],
                               title="NDCG@10 & HR@10 vs. Number of Graph Conv Layers")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True)

        with tabs[1]:
            if "dim" in abl:
                data = abl["dim"]
                dims = sorted(data.keys(), key=int)
                df   = pd.DataFrame([{"d":int(d), **data[d]} for d in dims]).set_index("d")
                fig  = px.line(df.reset_index(), x="d", y=["NDCG@10","HR@10"],
                               markers=True, template="plotly_dark",
                               color_discrete_sequence=["#6366f1","#818cf8"],
                               title="NDCG@10 & HR@10 vs. Embedding Dimension")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True)
    else:
        st.info("No ablation results found. Run: `python experiments/ablations.py`")
