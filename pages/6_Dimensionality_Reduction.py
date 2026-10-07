# ============================================================
# PatrolIQ - Dimensionality Reduction
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_reduced_data


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Dimensionality Reduction",
    page_icon="📉",
    layout="wide"
)


# ============================================================
# 2. Project Paths
# ============================================================

PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

REDUCED_PATH = DATA_DIR / "chicago_crime_reduced.parquet"
TSNE_PATH = DATA_DIR / "chicago_crime_tsne_sample.csv"
VARIANCE_PATH = DATA_DIR / "pca_explained_variance.csv"
IMPORTANCE_PATH = DATA_DIR / "pca_feature_importance.csv"
METHODOLOGY_PATH = DATA_DIR / "pca_methodology_summary.csv"
RECONSTRUCTION_PATH = DATA_DIR / "pca_reconstruction_metrics.csv"


# ============================================================
# 3. Data Sources
# ============================================================

HF_BASE_URL = "https://huggingface.co/datasets/AthiyamanP/PatrolIQ-Chicago-Crime-Analytics/resolve/main/"

def data_source(local_path):
    return local_path if local_path.exists() else HF_BASE_URL + local_path.name


# ============================================================
# 4. Load Data
# ============================================================

@st.cache_resource(show_spinner="Loading dimensionality-reduction data...")
def load_dimensionality_data():
    reduced = load_reduced_data()
    tsne = pd.read_csv(data_source(TSNE_PATH), low_memory=False)
    variance = pd.read_csv(data_source(VARIANCE_PATH))
    importance = pd.read_csv(data_source(IMPORTANCE_PATH))
    methodology = pd.read_csv(data_source(METHODOLOGY_PATH))
    reconstruction = pd.read_csv(data_source(RECONSTRUCTION_PATH))
    return reduced, tsne, variance, importance, methodology, reconstruction


try:

    (
        reduced_df,
        tsne_df,
        variance_df,
        importance_df,
        methodology_df,
        reconstruction_df
    ) = load_dimensionality_data()

except Exception as error:

    st.error(
        "Unable to load the dimensionality-reduction results."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 5. Validate PCA Columns
# ============================================================

required_pca_columns = [
    "PC1",
    "PC2",
    "PC3"
]

missing_pca = [
    column
    for column in required_pca_columns
    if column not in reduced_df.columns
]

if missing_pca:
    st.error(
        "Missing PCA columns in chicago_crime_reduced.parquet: "
        + ", ".join(missing_pca)
    )
    st.stop()


# ============================================================
# 6. Detect t-SNE Columns
# ============================================================

tsne_x_candidates = [
    "TSNE1",
    "TSNE_1",
    "tSNE1",
    "tSNE_1",
    "tsne1",
    "tsne_1"
]

tsne_y_candidates = [
    "TSNE2",
    "TSNE_2",
    "tSNE2",
    "tSNE_2",
    "tsne2",
    "tsne_2"
]


TSNE_X = next(
    (
        column
        for column in tsne_x_candidates
        if column in tsne_df.columns
    ),
    None
)

TSNE_Y = next(
    (
        column
        for column in tsne_y_candidates
        if column in tsne_df.columns
    ),
    None
)


# ============================================================
# 7. Page Header
# ============================================================

st.title("📉 Dimensionality Reduction")

st.markdown(
    """
    Explore the high-dimensional Chicago crime feature space using
    Principal Component Analysis (PCA) and t-SNE. PatrolIQ reduces
    the engineered analytical dimensions into visual representations
    that help reveal geographic, temporal, and crime-pattern structure.
    """
)


# ============================================================
# 8. Main Metrics
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Original Dimensions",
        "22"
    )

with col2:
    st.metric(
        "Principal Components",
        "3"
    )

with col3:
    st.metric(
        "Explained Variance",
        "71.77%"
    )

with col4:
    st.metric(
        "t-SNE Sample",
        f"{len(tsne_df):,}"
    )


st.caption(
    "PCA reduced 22 engineered analytical dimensions to three "
    "principal components explaining 71.77% of total variance."
)

st.divider()


# ============================================================
# 9. Sidebar Controls
# ============================================================

st.sidebar.header(
    "Dimensionality Reduction Controls"
)


available_color_options = []

candidate_color_columns = [
    "KMeans_Cluster",
    "Primary Type",
    "Temporal_Cluster",
    "District"
]

for column in candidate_color_columns:

    if column in reduced_df.columns:
        available_color_options.append(column)


if available_color_options:

    selected_color = st.sidebar.selectbox(
        "PCA Color Group",
        available_color_options,
        index=0
    )

else:

    selected_color = None


max_sample = min(
    30000,
    len(reduced_df)
)

default_sample = min(
    15000,
    max_sample
)


pca_sample_size = st.sidebar.slider(
    "PCA Interactive Sample Size",
    min_value=1000,
    max_value=max_sample,
    value=default_sample,
    step=1000
)


# ============================================================
# 10. PCA Interactive Sample
# ============================================================

pca_sample = reduced_df.sample(
    n=pca_sample_size,
    random_state=42
).copy()


if selected_color is not None:

    pca_sample[selected_color] = (
        pca_sample[selected_color]
        .astype(str)
    )


hover_columns = [
    column
    for column in [
        "Primary Type",
        "District",
        "Beat",
        "Community Area",
        "Hour",
        "Month",
        "Day_of_Week",
        "Crime_Severity_Score",
        "Arrest",
        "Domestic"
    ]
    if column in pca_sample.columns
]


# ============================================================
# 11. Interactive PCA 2D
# ============================================================

st.subheader(
    "Interactive PCA Visualization"
)


fig_pca2 = px.scatter(
    pca_sample,
    x="PC1",
    y="PC2",
    color=selected_color,
    hover_data=hover_columns,
    opacity=0.65,
    title="Interactive PCA 2D"
)


fig_pca2.update_traces(
    marker=dict(
        size=5
    )
)


fig_pca2.update_layout(
    height=650,
    legend_title_text=(
        selected_color
        if selected_color
        else "Group"
    )
)


st.plotly_chart(
    fig_pca2,
    width="stretch"
)


# ============================================================
# 12. Interactive PCA 3D
# ============================================================

st.subheader(
    "Interactive PCA 3D"
)


pca_3d_n = min(
    5000,
    len(pca_sample)
)


pca_3d = pca_sample.sample(
    n=pca_3d_n,
    random_state=42
).copy()


fig_pca3 = px.scatter_3d(
    pca_3d,
    x="PC1",
    y="PC2",
    z="PC3",
    color=selected_color,
    hover_data=hover_columns,
    opacity=0.7,
    title="Interactive PCA 3D"
)


fig_pca3.update_traces(
    marker=dict(
        size=3
    )
)


fig_pca3.update_layout(
    height=750,
    legend_title_text=(
        selected_color
        if selected_color
        else "Group"
    )
)


st.plotly_chart(
    fig_pca3,
    width="stretch"
)


st.caption(
    f"PCA 2D displays {len(pca_sample):,} records. "
    f"PCA 3D displays {len(pca_3d):,} records to maintain "
    "responsive browser interaction."
)


# ============================================================
# 13. Explained Variance
# ============================================================

st.divider()

st.subheader(
    "PCA Explained Variance"
)


# Use the exported file directly.
variance_plot = variance_df.copy()


# Detect likely columns from the exported file.
component_column = None
variance_column = None


for column in variance_plot.columns:

    lower = column.lower()

    if (
        component_column is None
        and "component" in lower
    ):
        component_column = column

    if (
        variance_column is None
        and "variance" in lower
        and "cumulative" not in lower
    ):
        variance_column = column


if (
    component_column is not None
    and variance_column is not None
):

    fig_variance = px.bar(
        variance_plot,
        x=component_column,
        y=variance_column,
        text=variance_column,
        title="Explained Variance by Principal Component"
    )


    fig_variance.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    st.plotly_chart(
        fig_variance,
        width="stretch"
    )

else:

    st.dataframe(
        variance_plot,
        width="stretch",
        hide_index=True
    )


col1, col2, col3 = st.columns(3)


col1.metric(
    "PC1",
    "29.68%"
)

col2.metric(
    "PC2",
    "28.38%"
)

col3.metric(
    "PC3",
    "13.71%"
)


st.success(
    "The first three principal components retain approximately "
    "71.77% of the variance in the engineered analytical feature space."
)


# ============================================================
# 14. PCA Feature Importance / Loadings
# ============================================================

st.divider()

st.subheader(
    "PCA Feature Importance"
)


st.markdown(
    """
    PCA loadings indicate which engineered features contribute most
    strongly to the principal-component representation.
    """
)


importance_plot = importance_df.copy()


# Try to identify feature and importance columns.
feature_column = None
importance_column = None


for column in importance_plot.columns:

    lower = column.lower()

    if (
        feature_column is None
        and "feature" in lower
    ):
        feature_column = column

    if (
        importance_column is None
        and (
            "importance" in lower
            or "loading" in lower
        )
    ):
        importance_column = column


if (
    feature_column is not None
    and importance_column is not None
):

    top_features = (
        importance_plot
        .sort_values(
            importance_column,
            ascending=False
        )
        .head(10)
    )


    fig_importance = px.bar(
        top_features,
        x=importance_column,
        y=feature_column,
        orientation="h",
        title="Top PCA Feature Contributions"
    )


    fig_importance.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )


    st.plotly_chart(
        fig_importance,
        width="stretch"
    )

else:

    st.dataframe(
        importance_plot,
        width="stretch",
        hide_index=True
    )


with st.expander(
    "View PCA feature importance table"
):

    st.dataframe(
        importance_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# 15. t-SNE Visualization
# ============================================================

st.divider()

st.subheader(
    "Interactive t-SNE Visualization"
)


if TSNE_X is None or TSNE_Y is None:

    st.warning(
        "The t-SNE coordinate column names could not be "
        "identified automatically."
    )

    st.dataframe(
        tsne_df.head(),
        width="stretch"
    )

else:

    tsne_color_options = [
        column
        for column in [
            "KMeans_Cluster",
            "Primary Type",
            "Temporal_Cluster",
            "District"
        ]
        if column in tsne_df.columns
    ]


    if tsne_color_options:

        tsne_color = st.selectbox(
            "Color t-SNE by",
            tsne_color_options,
            index=0
        )

        tsne_plot = tsne_df.copy()

        tsne_plot[tsne_color] = (
            tsne_plot[tsne_color]
            .astype(str)
        )

    else:

        tsne_color = None
        tsne_plot = tsne_df.copy()


    tsne_hover = [
        column
        for column in [
            "Primary Type",
            "District",
            "KMeans_Cluster",
            "Temporal_Cluster",
            "Hour",
            "Month",
            "Crime_Severity_Score"
        ]
        if column in tsne_plot.columns
    ]


    # Keep browser interaction responsive.
    interactive_tsne_n = min(
        10000,
        len(tsne_plot)
    )


    interactive_tsne = tsne_plot.sample(
        n=interactive_tsne_n,
        random_state=42
    )


    fig_tsne = px.scatter(
        interactive_tsne,
        x=TSNE_X,
        y=TSNE_Y,
        color=tsne_color,
        hover_data=tsne_hover,
        opacity=0.7,
        title="Interactive t-SNE 2D"
    )


    fig_tsne.update_traces(
        marker=dict(
            size=5
        )
    )


    fig_tsne.update_layout(
        height=700,
        legend_title_text=(
            tsne_color
            if tsne_color
            else "Group"
        )
    )


    st.plotly_chart(
        fig_tsne,
        width="stretch"
    )


    st.caption(
        f"Interactive visualization uses "
        f"{interactive_tsne_n:,} records from the "
        f"{len(tsne_df):,}-record t-SNE sample."
    )


# ============================================================
# 16. PCA vs t-SNE
# ============================================================

st.divider()

st.subheader(
    "PCA and t-SNE Comparison"
)


comparison_df = pd.DataFrame(
    {
        "Method": [
            "PCA",
            "t-SNE"
        ],

        "Primary Purpose": [
            "Global variance-preserving dimensionality reduction",
            "Non-linear local structure visualization"
        ],

        "Output Dimensions": [
            "3",
            "2"
        ],

        "Evaluation": [
            "71.77% explained variance",
            "KL divergence: 1.075575"
        ],

        "PatrolIQ Use": [
            "Feature compression, interpretation, and 2D/3D visualization",
            "Visual exploration of local crime-pattern structure"
        ]
    }
)


st.dataframe(
    comparison_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# 17. Reconstruction Metrics
# ============================================================

st.subheader(
    "PCA Reconstruction Quality"
)


metric_values = {}


if not reconstruction_df.empty:

    # Support either:
    # Metric | Value
    # or a one-row wide table.
    if (
        "Metric" in reconstruction_df.columns
        and "Value" in reconstruction_df.columns
    ):

        metric_values = dict(
            zip(
                reconstruction_df["Metric"],
                reconstruction_df["Value"]
            )
        )

    elif len(reconstruction_df) >= 1:

        metric_values = (
            reconstruction_df
            .iloc[0]
            .to_dict()
        )


def find_metric(
    dictionary,
    keyword,
    fallback
):

    for key, value in dictionary.items():

        if keyword.lower() in str(key).lower():

            try:
                return float(value)

            except (TypeError, ValueError):
                pass

    return fallback


mse_value = find_metric(
    metric_values,
    "mse",
    0.119569
)

rmse_value = find_metric(
    metric_values,
    "rmse",
    0.345787
)

mae_value = find_metric(
    metric_values,
    "mae",
    0.257404
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Reconstruction MSE",
    f"{mse_value:.6f}"
)

col2.metric(
    "Reconstruction RMSE",
    f"{rmse_value:.6f}"
)

col3.metric(
    "Reconstruction MAE",
    f"{mae_value:.6f}"
)


# ============================================================
# 18. Methodology
# ============================================================

with st.expander(
    "Dimensionality Reduction Methodology"
):

    st.markdown(
        """
        PatrolIQ uses 22 engineered analytical dimensions representing
        geographic, temporal, crime-category, and incident attributes.

        RobustScaler was applied before PCA to reduce sensitivity to
        extreme values. Three principal components were retained,
        explaining approximately 71.77% of total variance.

        The engineered feature space contains both raw and transformed
        representations of some underlying information. Therefore,
        the 22 dimensions should be interpreted as analytical features,
        not as 22 completely independent information sources.

        t-SNE is used as a complementary non-linear visualization
        technique for exploring local structure. It is not used as the
        PatrolIQ deployment clustering model.
        """
    )


    st.dataframe(
        methodology_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# 19. Interpretation
# ============================================================

st.info(
    """
    PCA provides a compact representation of the PatrolIQ feature
    space while retaining 71.77% of its variance in three components.
    PC1 primarily captures geographic structure, PC2 emphasizes
    temporal structure, and PC3 contributes additional crime-pattern
    information. t-SNE complements PCA by revealing local nonlinear
    structure in two dimensions. These visualizations support
    exploratory analysis and should not be interpreted as predictions
    of individual crime events.
    """
)