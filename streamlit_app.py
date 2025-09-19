import os
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = os.path.join("netflix_titles.csv", "netflix_titles.csv")

# ---------- App Config & Theming ----------
try:
    st.set_page_config(
        page_title="Netflix Explorer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except TypeError:
    # Fallback for older Streamlit versions without page_title argument
    st.set_page_config(
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

DARK_RED = "#E50914"
BLACK = "#141414"
DARK_GRAY = "#181818"
LIGHT_GRAY = "#B3B3B3"

CUSTOM_CSS = f"""
<style>
	/* Backgrounds */
	.stApp {{ background-color: {BLACK}; }}
	section[data-testid="stSidebar"] {{ background-color: {DARK_GRAY}; }}

	/* Text */
	h1, h2, h3, h4, h5, h6, p, span, label, div {{ color: #FFFFFF !important; }}

	/* Accents */
	.css-1v0mbdj, .st-emotion-cache-16idsys p, .st-emotion-cache-10trblm p {{ color: #FFFFFF !important; }}
	.stButton>button {{ background-color: {DARK_RED}; color: white; border: 0; }}
	.stDownloadButton>button {{ background-color: {DARK_RED}; color: white; border: 0; }}

	/* Cards */
	.block-container {{ padding-top: 1rem; }}
	.metric-card {{ background: {DARK_GRAY}; padding: 1rem; border-radius: 8px; border: 1px solid #2a2a2a; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    # Parse dates
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    # Release year numeric
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    # Critical non-nulls
    keep_cols = [c for c in ["type", "title"] if c in df.columns]
    df = df.dropna(subset=keep_cols)
    # Fill text fields
    for col in ["director", "cast", "country", "rating"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    if "listed_in" in df.columns:
        df["listed_in"] = df["listed_in"].fillna("Unknown")
    # Drop duplicates
    if "show_id" in df.columns:
        df = df.sort_values("show_id").drop_duplicates("show_id", keep="first")
    else:
        df = df.drop_duplicates()
    return df


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    if "listed_in" not in df.columns:
        return pd.DataFrame(columns=["genre"])
    genres = (
        df["listed_in"].str.split(",").explode().str.strip().replace({"": np.nan}).dropna()
    )
    return genres.to_frame("genre")


def simulate_viewing(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate viewing metadata: episodes, first/last watched dates.
	- Movies: episodes = 1
	- TV Shows: episodes = randint(4..40)
	- first_watched <= last_watched; both within the last 5 years
	"""
    rng = np.random.default_rng(seed)
    n = len(df)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=5)

    is_show = df["type"].eq("TV Show") if "type" in df.columns else pd.Series(False, index=df.index)
    episodes = np.where(is_show, rng.integers(4, 41, size=n), 1)
    first_days = rng.integers(0, (end - start).days + 1, size=n)
    first_watched = start + pd.to_timedelta(first_days, unit="D")
    last_additional = rng.integers(0, 120, size=n)
    last_watched = first_watched + pd.to_timedelta(last_additional, unit="D")
    last_watched = pd.Series(last_watched).clip(upper=end).values

    meta = pd.DataFrame({
        "episodes": episodes,
        "first_watched": pd.to_datetime(first_watched),
        "last_watched": pd.to_datetime(last_watched),
    }, index=df.index)
    return pd.concat([df, meta], axis=1)


def make_gauge(title: str, value: int, color: str = DARK_RED) -> go.Figure:
    return go.Figure(
        data=[go.Indicator(
            mode="gauge+number",
            value=int(value),
            title={"text": f"<b>{title}</b>", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, max(1, int(value) * 1.1)], "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "#222",
                "borderwidth": 1,
                "bordercolor": "#333",
            },
            number={"font": {"color": "white"}},
        )]
    ).update_layout(margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor=BLACK, plot_bgcolor=BLACK)


# ---------- Sidebar Controls ----------
st.sidebar.title("Filters")

# Dataset source
source = st.sidebar.radio("Data source", ["Bundled CSV"], index=0)

# Load & clean
try:
    df_raw = load_data(DATA_PATH)
    df = clean_data(df_raw)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Genre filter
genres_df = explode_genres(df)
available_genres = ["All"] + sorted(genres_df["genre"].unique().tolist()) if not genres_df.empty else ["All"]
selected_genres = st.sidebar.multiselect("Genre", options=available_genres, default=["All"])

# Type filter
available_types = sorted(df["type"].dropna().unique().tolist()) if "type" in df.columns else []
selected_types = st.sidebar.multiselect("Type", options=available_types, default=available_types)

# Year range
if "release_year" in df.columns and df["release_year"].notna().any():
    min_year, max_year = int(df["release_year"].min()), int(df["release_year"].max())
    year_range = st.sidebar.slider("Release year range", min_year, max_year, (min_year, max_year))
else:
    year_range = None

# Apply filters
mask = pd.Series(True, index=df.index)
if selected_types:
    mask &= df["type"].isin(selected_types)
if selected_genres and "All" not in selected_genres and not genres_df.empty:
    mask &= df["listed_in"].fillna("").apply(lambda s: any(g in s for g in selected_genres))
if year_range and "release_year" in df.columns:
    mask &= df["release_year"].between(year_range[0], year_range[1])

df_filt = df[mask].copy()

# Simulate viewing meta
view_df = simulate_viewing(df_filt)

# ---------- Header ----------
st.markdown(f"<h1>Netflix Explorer</h1>", unsafe_allow_html=True)
st.caption("Interactive dashboard for movies and TV shows")

# ---------- KPIs (Gauges) ----------
col1, col2, col3 = st.columns(3)
with col1:
    movies_count = int((view_df["type"] == "Movie").sum()) if "type" in view_df.columns else 0
    st.plotly_chart(make_gauge("Movies", movies_count), use_container_width=True, theme=None)
with col2:
    shows_count = int((view_df["type"] == "TV Show").sum()) if "type" in view_df.columns else 0
    st.plotly_chart(make_gauge("Series", shows_count), use_container_width=True, theme=None)
with col3:
    total_episodes = int(view_df.get("episodes", pd.Series([], dtype=int)).sum())
    st.plotly_chart(make_gauge("Episodes", total_episodes), use_container_width=True, theme=None)

# ---------- Pie: Movies vs Series ----------
if "type" in view_df.columns and not view_df.empty:
    pie_counts = view_df["type"].value_counts().reset_index()
    pie_counts.columns = ["type", "count"]
    fig_pie = px.pie(
        pie_counts, values="count", names="type",
        title="Movies vs Series",
        color="type",
        color_discrete_map={"Movie": DARK_RED, "TV Show": "#9b111e"},
    )
    fig_pie.update_traces(textfont_color="white")
    fig_pie.update_layout(paper_bgcolor=BLACK, plot_bgcolor=BLACK, font_color="white")
    st.plotly_chart(fig_pie, use_container_width=True, theme=None)

# ---------- Titles added per year ----------
if "date_added" in df.columns and df["date_added"].notna().any():
    added_year = df_filt.copy()
    added_year["year_added"] = added_year["date_added"].dt.year
    y_counts = added_year["year_added"].value_counts().sort_index()
    fig_year = px.bar(
        y_counts, x=y_counts.index, y=y_counts.values,
        title="Titles Added per Year",
        labels={"x": "Year", "y": "Count"},
    )
    fig_year.update_layout(paper_bgcolor=BLACK, plot_bgcolor=BLACK, font_color="white",
                           xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                           shapes=[dict(type="line", x0=0, x1=1, y0=0, y1=0, xref="paper", yref="paper",
                                        line=dict(color="#333"))])
    fig_year.update_traces(marker_color=DARK_RED)
    st.plotly_chart(fig_year, use_container_width=True, theme=None)

# ---------- Distribution by genre ----------
if not genres_df.empty:
    gmask = genres_df.index.isin(df_filt.index)
    genre_counts = genres_df[gmask]["genre"].value_counts().head(15)
    fig_genre = px.bar(
        x=genre_counts.values[::-1], y=genre_counts.index[::-1], orientation="h",
        title="Top Genres",
        labels={"x": "Count", "y": "Genre"}
    )
    fig_genre.update_traces(marker_color="#8b0000")
    fig_genre.update_layout(paper_bgcolor=BLACK, plot_bgcolor=BLACK, font_color="white",
                            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
    st.plotly_chart(fig_genre, use_container_width=True, theme=None)

# ---------- Viewing trend by weekday/month/year (simulated) ----------
if view_df["first_watched"].notna().any():
    vd = view_df.copy()
    vd["weekday"] = vd["first_watched"].dt.day_name()
    vd["month"] = vd["first_watched"].dt.month_name()
    vd["year"] = vd["first_watched"].dt.year

    colw1, colw2, colw3 = st.columns(3)
    with colw1:
        wd_counts = vd["weekday"].value_counts()[
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
        fig_wd = px.bar(wd_counts, x=wd_counts.index, y=wd_counts.values, title="Viewing by Weekday")
        fig_wd.update_traces(marker_color=DARK_RED)
        fig_wd.update_layout(paper_bgcolor=BLACK, plot_bgcolor=BLACK, font_color="white",
                             xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
        st.plotly_chart(fig_wd, use_container_width=True, theme=None)
    with colw2:
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        m_counts = vd["month"].value_counts().reindex(month_order, fill_value=0)
        fig_m = px.bar(m_counts, x=m_counts.index, y=m_counts.values, title="Viewing by Month")
        fig_m.update_traces(marker_color="#8b0000")
        fig_m.update_layout(paper_bgcolor=BLACK, plot_bgcolor=BLACK, font_color="white",
                            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
        st.plotly_chart(fig_m, use_container_width=True, theme=None)
    with colw3:
        y_counts2 = vd["year"].value_counts().sort_index()
        fig_y = px.bar(y_counts2, x=y_counts2.index, y=y_counts2.values, title="Viewing by Year")
        fig_y.update_traces(marker_color="#9b111e")
        fig_y.update_layout(paper_bgcolor=BLACK, plot_bgcolor=BLACK, font_color="white",
                            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"))
        st.plotly_chart(fig_y, use_container_width=True, theme=None)

# ---------- Top series by episodes (simulated) ----------
if "type" in view_df.columns and not view_df.empty:
    top_series = (
        view_df[view_df["type"] == "TV Show"][["title", "episodes", "first_watched", "last_watched", "listed_in"]]
        .sort_values("episodes", ascending=False)
        .head(15)
    )
    st.subheader("Top Series by Episodes (Simulated)")
    st.dataframe(top_series, use_container_width=True)

# ---------- Detailed table and CSV download ----------
st.subheader("Titles (Filtered)")
detail_cols = [c for c in
               ["title", "type", "release_year", "rating", "listed_in", "episodes", "first_watched", "last_watched"] if
               c in view_df.columns]
st.dataframe(view_df[detail_cols].sort_values(["type", "title"]).reset_index(drop=True), use_container_width=True)

csv_buf = io.StringIO()
view_df[detail_cols].to_csv(csv_buf, index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv_buf.getvalue(),
    file_name="netflix_filtered.csv",
    mime="text/csv",
)
