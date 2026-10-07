# ============================================================
# PatrolIQ - Smart Safety Analytics Platform
# Streamlit Main Application
# ============================================================

from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import load_clustered_data


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. Project Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chicago_crime_clustered.parquet"
)


# ============================================================
# 3. Load Dataset
# ============================================================

try:

    df = load_clustered_data()

    required_columns = [
        "Primary Type",
        "Arrest",
        "Domestic",
        "Latitude",
        "Longitude",
        "Hour",
        "Month",
        "KMeans_Cluster",
        "DBSCAN_Cluster",
        "Temporal_Cluster"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

        st.stop()

except Exception as error:

    st.error(
        "Unable to load the PatrolIQ dataset."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 5. Sidebar
# ============================================================

with st.sidebar:
    st.title("🚓 PatrolIQ")
    st.caption("Smart Safety Analytics Platform")
    st.divider()
    st.markdown(
        "Use the Streamlit page navigation above to explore "
        "crime analytics, temporal patterns, geographic hotspots, "
        "clustering, dimensionality reduction, and model performance."
    )
    st.divider()
    st.caption(f"Dataset: {len(df):,} crime records")


# ============================================================
# 6. Home / Overview
# ============================================================

st.title("🚓 PatrolIQ")
st.subheader("Smart Safety Analytics Platform")

st.markdown(
    """
    PatrolIQ is an interactive analytics platform for exploring recent
    Chicago crime records through descriptive analysis, geographic hotspot
    clustering, temporal pattern discovery, dimensionality reduction, and
    model-performance evaluation.
    """
)

st.divider()

st.subheader("Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Crime Records", f"{len(df):,}")

with col2:
    st.metric("Crime Types", f"{df['Primary Type'].nunique():,}")

with col3:
    st.metric("Geographic Hotspots", f"{df['KMeans_Cluster'].nunique():,}")

with col4:
    st.metric("Temporal Patterns", f"{df['Temporal_Cluster'].nunique():,}")

st.divider()

st.subheader("Analytics Pipeline")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Geographic K-Means", "K = 9")
    st.caption("Silhouette Score: 0.5012")

with col2:
    st.metric("Temporal K-Means", "K = 4")
    st.caption("Silhouette Score: 0.2702")

with col3:
    st.metric("PCA", "22 → 3")
    st.caption("Explained Variance: 71.77%")

with col4:
    st.metric("t-SNE Sample", "20,000")
    st.caption("KL Divergence: 1.075575")

st.success(
    "Geographic K-Means with K=9 is used as the PatrolIQ deployment "
    "clustering model because it provides scalable hotspot segmentation "
    "across the complete 500,000-record analytical dataset with a "
    "Silhouette Score above 0.50."
)

st.divider()

st.subheader("Explore PatrolIQ")

st.markdown(
    """
    Use the page navigation in the sidebar to explore:

    - **Crime Dashboard** — crime categories, arrest patterns, domestic incidents, and distributions.
    - **Temporal Analysis** — hourly, weekday, monthly, seasonal, and temporal-cluster patterns.
    - **Crime Map** — geographic crime distribution, heatmaps, and K-Means hotspot zones.
    - **Clustering Results** — K-Means, DBSCAN, hierarchical, and temporal clustering results.
    - **Dimensionality Reduction** — interactive PCA and t-SNE visualizations and feature importance.
    - **Model Performance** — clustering metrics, dimensionality-reduction metrics, MLflow runs, and registered models.
    """
)

st.divider()

st.subheader("Dataset Preview")

preview_columns = [
    "Date",
    "Primary Type",
    "Description",
    "District",
    "Latitude",
    "Longitude",
    "KMeans_Cluster",
    "Temporal_Cluster",
]

available_preview_columns = [
    column for column in preview_columns if column in df.columns
]

st.dataframe(
    df[available_preview_columns].head(10),
    width="stretch",
    hide_index=True,
)

st.info(
    "PatrolIQ uses 500,000 recent cleaned Chicago crime records for the "
    "analytical workflow presented in this application."
)
