import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Performance | PatrolIQ",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

DR_METRICS_FILE = DATA_DIR / "dimensionality_reduction_metrics.csv"
PCA_VARIANCE_FILE = DATA_DIR / "pca_explained_variance.csv"
PCA_RECON_FILE = DATA_DIR / "pca_reconstruction_metrics.csv"

HF_BASE_URL = "https://huggingface.co/datasets/AthiyamanP/PatrolIQ-Chicago-Crime-Analytics/resolve/main/"

def read_project_csv(local_path):
    source = local_path if local_path.exists() else HF_BASE_URL + local_path.name
    return pd.read_csv(source)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Model Performance")

st.write(
    "Monitor the performance of PatrolIQ clustering and dimensionality "
    "reduction models and review the models tracked through the MLflow "
    "experiment workflow."
)


# ============================================================
# CLUSTERING PERFORMANCE DATA
# ============================================================

clustering_metrics = pd.DataFrame(
    {
        "Model": [
            "Geographic K-Means",
            "DBSCAN",
            "Hierarchical",
            "Temporal K-Means"
        ],
        "Configuration": [
            "K = 9",
            "1,049 clusters",
            "K = 10",
            "K = 4"
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
        ],
        "Davies-Bouldin Index": [
            0.7153,
            None,
            0.6240,
            1.1319
        ]
    }
)


# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

st.header("Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Geographic K-Means",
        "0.5012",
        help="Silhouette Score on the complete 500,000-record analytical dataset."
    )

with col2:
    st.metric(
        "Hierarchical",
        "0.5170",
        help="Silhouette Score on a reproducible 10,000-record sample."
    )

with col3:
    st.metric(
        "PCA Explained Variance",
        "71.77%"
    )

with col4:
    st.metric(
        "t-SNE KL Divergence",
        "1.075575"
    )

st.caption(
    "Clustering metrics were evaluated on different scopes where required. "
    "Hierarchical clustering was evaluated on a 10,000-record sample and "
    "therefore should not be treated as a direct full-data comparison with "
    "Geographic K-Means."
)

st.divider()


# ============================================================
# CLUSTERING MODEL COMPARISON
# ============================================================

st.header("Clustering Model Performance")

st.dataframe(
    clustering_metrics,
    width="stretch",
    hide_index=True
)


# ============================================================
# SILHOUETTE SCORE COMPARISON
# ============================================================

st.subheader("Silhouette Score Comparison")

silhouette_fig = px.bar(
    clustering_metrics,
    x="Model",
    y="Silhouette Score",
    text="Silhouette Score",
    title="Clustering Silhouette Scores"
)

silhouette_fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

silhouette_fig.add_hline(
    y=0.50,
    line_dash="dash",
    annotation_text="0.50 Reference",
    annotation_position="top right"
)

silhouette_fig.update_layout(
    xaxis_title="Model",
    yaxis_title="Silhouette Score",
    height=500
)

st.plotly_chart(
    silhouette_fig,
    width="stretch"
)

st.info(
    "Geographic K-Means achieved a Silhouette Score of 0.5012 on the "
    "complete 500,000-record dataset. Hierarchical clustering produced "
    "0.5170 on a 10,000-record sample. DBSCAN produced a negative "
    "silhouette score on the complete dataset, indicating that its "
    "full-data clustering structure was less suitable for the PatrolIQ "
    "hotspot representation."
)


# ============================================================
# DAVIES-BOULDIN COMPARISON
# ============================================================

st.subheader("Davies-Bouldin Index Comparison")

db_data = clustering_metrics.dropna(
    subset=["Davies-Bouldin Index"]
).copy()

db_fig = px.bar(
    db_data,
    x="Model",
    y="Davies-Bouldin Index",
    text="Davies-Bouldin Index",
    title="Davies-Bouldin Index by Model"
)

db_fig.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

db_fig.update_layout(
    xaxis_title="Model",
    yaxis_title="Davies-Bouldin Index",
    height=500
)

st.plotly_chart(
    db_fig,
    width="stretch"
)

st.caption(
    "Lower Davies-Bouldin Index values indicate more compact and "
    "better-separated clusters. Evaluation scope should also be considered "
    "when comparing these values."
)

st.divider()


# ============================================================
# DEPLOYMENT MODEL
# ============================================================

st.header("Deployment Model Selection")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected Model",
        "Geographic K-Means"
    )

with col2:
    st.metric(
        "Geographic Clusters",
        "9"
    )

with col3:
    st.metric(
        "Evaluation Records",
        "500,000"
    )

st.success(
    "Geographic K-Means with K=9 is used as the PatrolIQ geographic "
    "deployment clustering model. It combines full-dataset scalability, "
    "interpretable geographic zones, a Silhouette Score above 0.50, and "
    "practical hotspot segmentation for the Streamlit application."
)

st.divider()


# ============================================================
# DIMENSIONALITY REDUCTION PERFORMANCE
# ============================================================

st.header("Dimensionality Reduction Performance")


# ------------------------------------------------------------
# PCA SUMMARY
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Input Dimensions",
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
        "20,000"
    )


# ============================================================
# PCA EXPLAINED VARIANCE FILE
# ============================================================

if True:

    try:
        pca_variance = read_project_csv(PCA_VARIANCE_FILE)

        st.subheader("PCA Explained Variance")

        component_col = None
        variance_col = None

        for col in pca_variance.columns:

            col_lower = col.lower()

            if component_col is None and "component" in col_lower:
                component_col = col

            if (
                variance_col is None
                and "variance" in col_lower
                and "cumulative" not in col_lower
            ):
                variance_col = col

        if component_col and variance_col:

            plot_df = pca_variance.copy()

            if plot_df[variance_col].max() <= 1:
                plot_df[variance_col] = (
                    plot_df[variance_col] * 100
                )

            variance_fig = px.bar(
                plot_df,
                x=component_col,
                y=variance_col,
                title="Explained Variance by Principal Component"
            )

            variance_fig.update_layout(
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance (%)",
                height=500
            )

            st.plotly_chart(
                variance_fig,
                width="stretch"
            )

        else:

            st.dataframe(
                pca_variance,
                width="stretch",
                hide_index=True
            )

    except Exception as e:

        st.warning(
            f"PCA explained variance file could not be displayed: {e}"
        )


# ============================================================
# PCA RECONSTRUCTION METRICS
# ============================================================

st.subheader("PCA Reconstruction Quality")

reconstruction_values = {
    "MSE": 0.119569,
    "RMSE": 0.345787,
    "MAE": 0.257404
}

if True:

    try:

        recon_df = read_project_csv(PCA_RECON_FILE)

        numeric_values = {}

        for col in recon_df.columns:

            col_lower = col.lower()

            if "rmse" in col_lower:
                numeric_values["RMSE"] = float(
                    recon_df[col].iloc[0]
                )

            elif "mae" in col_lower:
                numeric_values["MAE"] = float(
                    recon_df[col].iloc[0]
                )

            elif "mse" in col_lower:
                numeric_values["MSE"] = float(
                    recon_df[col].iloc[0]
                )

        if numeric_values:
            reconstruction_values.update(
                numeric_values
            )

    except Exception:
        pass


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Reconstruction MSE",
        f"{reconstruction_values['MSE']:.6f}"
    )

with col2:
    st.metric(
        "Reconstruction RMSE",
        f"{reconstruction_values['RMSE']:.6f}"
    )

with col3:
    st.metric(
        "Reconstruction MAE",
        f"{reconstruction_values['MAE']:.6f}"
    )


# ============================================================
# PCA / t-SNE COMPARISON
# ============================================================

st.subheader("Dimensionality Reduction Comparison")

dr_comparison = pd.DataFrame(
    {
        "Method": [
            "PCA",
            "t-SNE"
        ],
        "Purpose": [
            "Global variance-preserving dimensionality reduction",
            "Non-linear local structure visualization"
        ],
        "Output Dimensions": [
            3,
            2
        ],
        "Evaluation": [
            "71.77% explained variance",
            "KL divergence = 1.075575"
        ],
        "PatrolIQ Use": [
            "Feature compression, interpretation and 2D/3D visualization",
            "Visual exploration of local crime-pattern structure"
        ]
    }
)

st.dataframe(
    dr_comparison,
    width="stretch",
    hide_index=True
)

st.divider()


# ============================================================
# MLFLOW TRACKING SUMMARY
# ============================================================

st.header("MLflow Experiment Tracking")

st.write(
    "PatrolIQ uses MLflow to track clustering and dimensionality reduction "
    "experiments, compare model metrics, and register deployment models."
)

mlflow_runs = pd.DataFrame(
    {
        "Experiment Run": [
            "Geographic K-Means K=9",
            "DBSCAN",
            "Hierarchical K=10",
            "Temporal K-Means K=4",
            "Dimensionality Reduction"
        ],
        "Run ID": [
            "ae8495d7e7204b7188b355abedeca188",
            "707f24859d2f4811801d8aa8162c7afa",
            "682d79346e474a1fbd118269d1951751",
            "c9a79842f77e48a091909b121860c534",
            "a69a61db856c41dfbdc971a6883e6e62"
        ],
        "Primary Metric": [
            "Silhouette = 0.5012",
            "Silhouette = -0.5818",
            "Silhouette = 0.5170",
            "Silhouette = 0.2702",
            "Explained Variance = 71.77%"
        ],
        "Evaluation Scope": [
            "500,000 records",
            "500,000 records",
            "10,000-record sample",
            "500,000 records",
            "500,000 records / t-SNE 20,000 sample"
        ]
    }
)

st.dataframe(
    mlflow_runs,
    width="stretch",
    hide_index=True
)


# ============================================================
# REGISTERED MODELS
# ============================================================

st.subheader("Registered Models")

registered_models = pd.DataFrame(
    {
        "Registered Model": [
            "PatrolIQ_Geographic_KMeans",
            "PatrolIQ_Temporal_KMeans"
        ],
        "Version": [
            1,
            1
        ],
        "Role": [
            "Geographic hotspot segmentation",
            "Temporal crime-pattern segmentation"
        ]
    }
)

st.dataframe(
    registered_models,
    width="stretch",
    hide_index=True
)


# ============================================================
# MLFLOW STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Tracked Pipeline Runs",
        "5"
    )

with col2:
    st.metric(
        "Registered Models",
        "2"
    )

with col3:
    st.metric(
        "Deployment Clusters",
        "9"
    )

st.info(
    "MLflow records the experiment parameters, evaluation metrics and model "
    "artifacts generated during the PatrolIQ modelling workflow. The "
    "geographic and temporal K-Means models were registered through the "
    "MLflow Model Registry."
)

st.divider()


# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================

st.header("Model Performance Summary")

summary = pd.DataFrame(
    {
        "Pipeline Component": [
            "Geographic Clustering",
            "Density Clustering",
            "Hierarchical Clustering",
            "Temporal Clustering",
            "PCA",
            "t-SNE"
        ],
        "Method": [
            "K-Means K=9",
            "DBSCAN",
            "Hierarchical K=10",
            "K-Means K=4",
            "PCA 22 → 3",
            "t-SNE 22 → 2"
        ],
        "Performance": [
            "Silhouette 0.5012 | DBI 0.7153",
            "Silhouette -0.5818 | 1,049 clusters",
            "Silhouette 0.5170 | DBI 0.6240",
            "Silhouette 0.2702 | DBI 1.1319",
            "71.77% explained variance",
            "KL divergence 1.075575"
        ],
        "Use in PatrolIQ": [
            "Deployment geographic hotspot model",
            "Density-structure comparison",
            "Sample-based clustering comparison",
            "Recurring temporal pattern analysis",
            "Dimensionality reduction and visualization",
            "Local structure visualization"
        ]
    }
)

st.dataframe(
    summary,
    width="stretch",
    hide_index=True
)


# ============================================================
# INTERPRETATION
# ============================================================

st.success(
    "PatrolIQ uses Geographic K-Means K=9 as its geographic deployment "
    "model because it provides scalable clustering across the complete "
    "500,000-record analytical dataset while maintaining a Silhouette "
    "Score above 0.50. PCA compresses the engineered 22-dimensional "
    "feature space into three principal components while retaining "
    "approximately 71.77% of total variance. MLflow provides experiment "
    "tracking and model registration for the modelling pipeline."
)