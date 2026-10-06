# ============================================================
# PatrolIQ - Clustering Results
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Clustering Results",
    page_icon="🧩",
    layout="wide"
)


# ============================================================
# 2. Project Paths
# ============================================================

PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chicago_crime_clustered.parquet"
)


# ============================================================
# 3. Load Dataset
# ============================================================

HF_DATA_URL = "https://huggingface.co/datasets/AthiyamanP/PatrolIQ-Chicago-Crime-Analytics/resolve/main/chicago_crime_clustered.parquet"


@st.cache_data(show_spinner="Loading PatrolIQ data...")
def load_data():
    source = DATA_PATH if DATA_PATH.exists() else HF_DATA_URL
    data = pd.read_parquet(source, engine='pyarrow')

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    return data


try:

    df = load_data()

except Exception as error:

    st.error(
        "Unable to load the PatrolIQ clustering dataset."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 4. Validate Required Columns
# ============================================================

required_columns = [
    "KMeans_Cluster",
    "DBSCAN_Cluster",
    "Temporal_Cluster",
    "Primary Type",
    "District",
    "Latitude",
    "Longitude",
    "Arrest",
    "Domestic"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "Required clustering columns are missing: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# 5. Page Header
# ============================================================

st.title("🧩 Clustering Results")

st.markdown(
    """
    Compare the clustering approaches used in PatrolIQ and explore
    the geographic hotspot zones produced by the deployment model.
    """
)


# ============================================================
# 6. Main Clustering Metrics
# ============================================================

st.subheader("Clustering Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "K-Means",
        "9 Clusters"
    )

    st.caption(
        "500,000 records"
    )


with col2:

    st.metric(
        "K-Means Silhouette",
        "0.5012"
    )

    st.caption(
        "Davies-Bouldin: 0.7153"
    )


with col3:

    st.metric(
        "DBSCAN",
        "1,049 Clusters"
    )

    st.caption(
        "Noise: 31,847 records"
    )


with col4:

    st.metric(
        "Hierarchical",
        "10 Clusters"
    )

    st.caption(
        "10,000-record sample"
    )


st.divider()


# ============================================================
# 7. Algorithm Comparison
# ============================================================

st.subheader("Clustering Algorithm Comparison")


comparison_df = pd.DataFrame(
    {
        "Algorithm": [
            "K-Means K=9",
            "DBSCAN",
            "Hierarchical K=10"
        ],

        "Evaluation Data": [
            "500,000 records",
            "500,000 records",
            "10,000-record sample"
        ],

        "Clusters": [
            9,
            1049,
            10
        ],

        "Silhouette Score": [
            0.5012,
            -0.5818,
            0.5170
        ],

        "Davies-Bouldin Index": [
            0.7153,
            None,
            0.6240
        ]
    }
)


st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 8. Silhouette Score Comparison
# ============================================================

fig_silhouette = px.bar(
    comparison_df,
    x="Algorithm",
    y="Silhouette Score",
    text="Silhouette Score",
    title="Silhouette Score Comparison"
)


fig_silhouette.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)


fig_silhouette.add_hline(
    y=0.5,
    line_dash="dash",
    annotation_text="0.50 Reference"
)


st.plotly_chart(
    fig_silhouette,
    use_container_width=True
)


st.caption(
    "Hierarchical clustering achieved the highest silhouette score "
    "on its 10,000-record sample. Because it was evaluated on a sample "
    "rather than the complete dataset, the value should not be treated "
    "as a direct full-data comparison with K-Means."
)


# ============================================================
# 9. Davies-Bouldin Comparison
# ============================================================

db_comparison = comparison_df[
    comparison_df["Davies-Bouldin Index"].notna()
].copy()


fig_db = px.bar(
    db_comparison,
    x="Algorithm",
    y="Davies-Bouldin Index",
    text="Davies-Bouldin Index",
    title="Davies-Bouldin Index Comparison"
)


fig_db.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)


st.plotly_chart(
    fig_db,
    use_container_width=True
)


st.caption(
    "For the Davies-Bouldin Index, lower values indicate better "
    "cluster separation and compactness."
)


# ============================================================
# 10. Deployment Model
# ============================================================

st.subheader("Deployment Model Selection")


st.success(
    """
    Geographic K-Means with K=9 is used as the PatrolIQ deployment
    clustering model. It provides scalable clustering across the complete
    500,000-record dataset, a Silhouette Score above 0.50, interpretable
    geographic centers, and practical hotspot segmentation for the
    Streamlit application.
    """
)


# ============================================================
# 11. K-Means Cluster Distribution
# ============================================================

st.subheader("K-Means Geographic Zone Distribution")


kmeans_counts = (
    df["KMeans_Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)


kmeans_counts.columns = [
    "Geographic Zone",
    "Crime Records"
]


kmeans_counts["Geographic Zone"] = (
    kmeans_counts["Geographic Zone"]
    .astype(str)
)


fig_kmeans = px.bar(
    kmeans_counts,
    x="Geographic Zone",
    y="Crime Records",
    color="Geographic Zone",
    title="Recorded Crime Volume by K-Means Geographic Zone"
)


st.plotly_chart(
    fig_kmeans,
    use_container_width=True
)


# ============================================================
# 12. K-Means Cluster Profiles
# ============================================================

st.subheader("K-Means Geographic Zone Profiles")


profile_rows = []


for cluster in sorted(
    df["KMeans_Cluster"]
    .dropna()
    .unique()
):

    cluster_df = df[
        df["KMeans_Cluster"] == cluster
    ]


    row = {
        "Geographic Zone": int(cluster),

        "Crime Records":
            len(cluster_df),

        "Share (%)":
            round(
                len(cluster_df)
                / len(df)
                * 100,
                2
            ),

        "Top Crime Type":
            cluster_df[
                "Primary Type"
            ].mode().iloc[0],

        "Top District":
            cluster_df[
                "District"
            ].mode().iloc[0],

        "Center Latitude":
            round(
                cluster_df[
                    "Latitude"
                ].mean(),
                4
            ),

        "Center Longitude":
            round(
                cluster_df[
                    "Longitude"
                ].mean(),
                4
            ),

        "Arrest Rate (%)":
            round(
                cluster_df[
                    "Arrest"
                ].mean()
                * 100,
                2
            ),

        "Domestic Rate (%)":
            round(
                cluster_df[
                    "Domestic"
                ].mean()
                * 100,
                2
            )
    }


    if "Crime_Severity_Score" in cluster_df.columns:

        row["Average Severity"] = round(
            cluster_df[
                "Crime_Severity_Score"
            ].mean(),
            2
        )


    profile_rows.append(row)


profile_df = pd.DataFrame(
    profile_rows
)


st.dataframe(
    profile_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 13. DBSCAN Analysis
# ============================================================

st.subheader("DBSCAN Density-Based Clustering")


dbscan_clusters = (
    df.loc[
        df["DBSCAN_Cluster"] != -1,
        "DBSCAN_Cluster"
    ]
    .nunique()
)


dbscan_noise = (
    df["DBSCAN_Cluster"] == -1
).sum()


dbscan_noise_pct = (
    dbscan_noise
    / len(df)
    * 100
)


largest_dbscan_cluster = (
    df.loc[
        df["DBSCAN_Cluster"] != -1,
        "DBSCAN_Cluster"
    ]
    .value_counts(
        normalize=True
    )
    .iloc[0]
    * 100
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "DBSCAN Clusters",
    f"{dbscan_clusters:,}"
)


col2.metric(
    "Noise Records",
    f"{dbscan_noise:,}"
)


col3.metric(
    "Noise Share",
    f"{dbscan_noise_pct:.2f}%"
)


col4.metric(
    "Largest Cluster Share",
    f"{largest_dbscan_cluster:.2f}%"
)


st.markdown(
    """
    DBSCAN identified many local density groups and explicitly
    separated noise records. On the complete analytical dataset,
    however, the large number of clusters and negative silhouette
    score made the result less suitable for the PatrolIQ deployment
    hotspot representation.
    """
)


# ============================================================
# 14. DBSCAN Largest Clusters
# ============================================================

dbscan_counts = (
    df[
        df["DBSCAN_Cluster"] != -1
    ]["DBSCAN_Cluster"]
    .value_counts()
    .head(15)
    .reset_index()
)


dbscan_counts.columns = [
    "DBSCAN Cluster",
    "Crime Records"
]


dbscan_counts["DBSCAN Cluster"] = (
    dbscan_counts["DBSCAN Cluster"]
    .astype(str)
)


fig_dbscan = px.bar(
    dbscan_counts,
    x="DBSCAN Cluster",
    y="Crime Records",
    title="15 Largest DBSCAN Clusters"
)


st.plotly_chart(
    fig_dbscan,
    use_container_width=True
)


# ============================================================
# 15. Hierarchical Clustering
# ============================================================

st.subheader("Hierarchical Clustering")


col1, col2, col3 = st.columns(3)


col1.metric(
    "Selected Clusters",
    "10"
)


col2.metric(
    "Silhouette Score",
    "0.5170"
)


col3.metric(
    "Davies-Bouldin Index",
    "0.6240"
)


st.info(
    """
    Hierarchical clustering was evaluated on a reproducible
    10,000-record sample because standard agglomerative clustering
    is computationally expensive for the complete 500,000-record
    dataset. K=10 produced the strongest sampled hierarchical result.
    """
)


# ============================================================
# 16. Geographic K-Means vs Temporal K-Means
# ============================================================

st.subheader("Geographic and Temporal Clustering")


col1, col2 = st.columns(2)


with col1:

    geographic_counts = (
        df["KMeans_Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    geographic_counts.columns = [
        "Cluster",
        "Crime Records"
    ]

    geographic_counts["Cluster"] = (
        geographic_counts["Cluster"]
        .astype(str)
    )


    fig_geo = px.bar(
        geographic_counts,
        x="Cluster",
        y="Crime Records",
        title="Geographic K-Means — K=9"
    )


    st.plotly_chart(
        fig_geo,
        use_container_width=True
    )


with col2:

    temporal_counts = (
        df["Temporal_Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    temporal_counts.columns = [
        "Cluster",
        "Crime Records"
    ]

    temporal_counts["Cluster"] = (
        temporal_counts["Cluster"]
        .astype(str)
    )


    fig_temporal = px.bar(
        temporal_counts,
        x="Cluster",
        y="Crime Records",
        title="Temporal K-Means — K=4"
    )


    st.plotly_chart(
        fig_temporal,
        use_container_width=True
    )


# ============================================================
# 17. Model Summary
# ============================================================

st.subheader("Clustering Summary")


summary_df = pd.DataFrame(
    {
        "Model": [
            "Geographic K-Means",
            "DBSCAN",
            "Hierarchical",
            "Temporal K-Means"
        ],

        "Purpose": [
            "Geographic hotspot segmentation",
            "Density-based geographic structure",
            "Hierarchical geographic structure",
            "Temporal behavior segmentation"
        ],

        "Clusters": [
            9,
            1049,
            10,
            4
        ],

        "Evaluation Scope": [
            "500,000 records",
            "500,000 records",
            "10,000-record sample",
            "500,000 records"
        ],

        "Silhouette Score": [
            0.5012,
            -0.5818,
            0.5170,
            0.2702
        ]
    }
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 18. Interpretation
# ============================================================

st.info(
    """
    The clustering algorithms serve different analytical purposes.
    K-Means K=9 is the deployment choice because it combines
    full-dataset scalability, interpretable geographic zones, and
    a Silhouette Score above 0.50. DBSCAN is retained as a
    density-based comparison, while hierarchical clustering provides
    strong sampled separation but is not used as the full-data
    deployment model. Temporal K-Means independently identifies
    four recurring temporal behavior patterns.
    """
)