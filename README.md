# 🚓 PatrolIQ – Smart Safety Analytics Platform

PatrolIQ is an end-to-end data science and analytics project for exploring recent Chicago crime records through data preprocessing, exploratory analysis, feature engineering, clustering, dimensionality reduction, experiment tracking, and an interactive Streamlit application.

## 🌐 Live Application

**Streamlit Cloud:**  
https://athiyamanomen-web-patroliq-smart-safety-analytics-app-h6hm31.streamlit.app/

## 📌 Project Overview

The project analyzes **500,000 recent cleaned Chicago crime records** and provides:

- Crime distribution and descriptive analytics
- Temporal crime-pattern analysis
- Geographic hotspot discovery
- K-Means, DBSCAN, and Hierarchical clustering comparison
- Temporal K-Means pattern discovery
- PCA and t-SNE dimensionality reduction
- MLflow experiment tracking and model registration
- Interactive multi-page Streamlit dashboards
- Cloud deployment using GitHub, Streamlit Community Cloud, and Hugging Face dataset hosting

## 🏗️ Architecture

```text
Chicago Crime Data
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Feature Engineering
        │
        ├──────────────► Exploratory Data Analysis
        │
        ▼
Clustering
(K-Means / DBSCAN / Hierarchical / Temporal K-Means)
        │
        ▼
Dimensionality Reduction
(PCA / t-SNE)
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Processed Deployment Data
(Parquet + CSV)
        │
        ▼
Hugging Face Dataset Repository
        │
        ▼
Streamlit Multi-Page Application
        │
        ▼
Streamlit Community Cloud
```

## 📊 Dataset

The analytical dataset contains **500,000 recent cleaned crime records**.

Key validated characteristics:

| Metric | Value |
|---|---:|
| Records | 500,000 |
| Crime Types | 31 |
| Police Districts | 23 |
| Community Areas | 77 |
| Beats | 274 |
| Missing Values | 0 |
| Duplicate Records | 0 |

The analyzed period spans approximately **May 2024 to May 2026**.

> The project brief referenced the complete historical Chicago crime dataset and 33 crime types. The available source used for this implementation contained 847,276 records, from which the 500,000 most recent cleaned records were selected. The resulting analytical sample contains 31 crime types. No categories were fabricated to force the expected count.

## 🧹 Data Preprocessing & Feature Engineering

The preprocessing pipeline includes:

- Missing-value and duplicate validation
- Date/time parsing
- Recent-record sampling
- Hour, day, month, weekend, season, and related temporal features
- Geographic coordinate features and coordinate binning
- Police administrative-area features
- Crime severity heuristic
- Crime/category frequency features
- Categorical encoding for analytical workflows
- Geographic feature scaling
- Data-quality validation

The crime severity score used in PatrolIQ is a **project-defined analytical heuristic** and should not be interpreted as an official Chicago Police Department severity classification.

## 🔍 Exploratory Data Analysis

Major observations from the 500,000-record analytical dataset include:

- **THEFT:** 116,381 records (23.28%)
- **Peak recorded hour:** 00:00
- **Peak recorded day:** Friday
- **Highest raw monthly count:** May
- **Arrest rate:** 15.05%
- **Domestic incident rate:** 18.84%

Temporal counts represent recorded incidents in the available data period and should not be interpreted as population-adjusted crime risk. Monthly and seasonal totals are also affected by the dataset's partial-year coverage.

## 🧩 Geographic Clustering

### K-Means

PatrolIQ uses Geographic K-Means as the deployment hotspot model.

| Metric | Result |
|---|---:|
| Records | 500,000 |
| Clusters | 9 |
| Silhouette Score | 0.5012 |
| Davies-Bouldin Index | 0.7153 |

The model uses:

- Latitude
- Longitude
- Beat
- District
- Community Area

K=9 provided the strongest balance of full-dataset clustering quality, scalability, and interpretable geographic hotspot segmentation.

### DBSCAN

| Metric | Result |
|---|---:|
| Records | 500,000 |
| Clusters | 1,049 |
| Noise Records | 31,847 |
| Noise Rate | 6.37% |
| Silhouette Score | -0.5818 |

DBSCAN identified many small density-based regions and noise points but was not selected as the deployment clustering model.

### Hierarchical Clustering

Hierarchical clustering was evaluated on a reproducible **10,000-record sample** because standard agglomerative clustering is not practical for the complete 500,000-record dataset.

| Metric | Result |
|---|---:|
| Sample | 10,000 |
| Clusters | 10 |
| Silhouette Score | 0.5170 |
| Davies-Bouldin Index | 0.6240 |

Although its sample-level silhouette score is slightly higher than K-Means, it should not be treated as a direct full-dataset comparison.

## ⏰ Temporal Clustering

Temporal K-Means identifies four recurring temporal crime patterns using:

- Hour
- Day of week
- Month

| Metric | Result |
|---|---:|
| Records | 500,000 |
| Temporal Clusters | 4 |
| Silhouette Score | 0.2702 |
| Davies-Bouldin Index | 1.1319 |

These clusters are used to explore similar time-of-occurrence behavior rather than geographic hotspots.

## 📉 Dimensionality Reduction

### Principal Component Analysis

The engineered analytical representation contains **22 features**.

PCA reduced:

**22 dimensions → 3 principal components**

with:

**71.77% cumulative explained variance**

The three components primarily capture geographic, temporal, and crime-pattern structure.

> Some engineered variables are mathematically related. For example, `DayOfYear_Normalized` is a rescaled representation of `Date_DayOfYear`. The 22 inputs therefore represent engineered analytical dimensions, not 22 fully independent information sources.

### t-SNE

t-SNE was applied to a **20,000-record sample** for nonlinear two-dimensional visualization.

| Metric | Result |
|---|---:|
| Sample Size | 20,000 |
| KL Divergence | 1.075575 |

The Streamlit application provides interactive PCA and t-SNE visualizations.

## 🧪 MLflow Experiment Tracking

MLflow is used to track the analytical pipeline, including:

- Geographic K-Means
- DBSCAN
- Hierarchical clustering
- Temporal K-Means
- PCA
- t-SNE

Tracked information includes model parameters, clustering metrics, explained variance, reconstruction metrics, and dimensionality-reduction metrics.

Registered models include:

- `PatrolIQ_Geographic_KMeans`
- `PatrolIQ_Temporal_KMeans`

## 🖥️ Streamlit Application

PatrolIQ provides seven application views:

1. **Home** – project overview and pipeline metrics
2. **Crime Dashboard** – crime distributions, arrest patterns, domestic incidents, and interactive filters
3. **Temporal Analysis** – hourly, weekday, monthly, seasonal, and temporal-cluster analysis
4. **Crime Map** – geographic density and K-Means hotspot-zone visualization
5. **Clustering Results** – clustering algorithm comparison and hotspot profiles
6. **Model Performance** – clustering and dimensionality-reduction performance
7. **Dimensionality Reduction** – interactive PCA and t-SNE exploration

The analytical clustering models remain based on the complete 500,000-record dataset. Visualization sampling is used where appropriate to keep interactive geographic and dimensionality-reduction plots responsive.

## ☁️ Deployment Architecture

Large deployment datasets are hosted separately from the GitHub code repository.

### GitHub

Repository:

https://github.com/athiyamanomen-web/PatrolIQ-Smart-Safety-Analytics

### Hugging Face

Dataset repository:

https://huggingface.co/datasets/AthiyamanP/PatrolIQ-Chicago-Crime-Analytics

Large 500,000-row deployment datasets are stored as **Parquet** files to reduce transfer size and improve loading performance while preserving all records.

Approximate conversion results:

| Dataset | CSV | Parquet | Reduction |
|---|---:|---:|---:|
| Clustered | 176.58 MB | 38.32 MB | 78.3% |
| PCA Reduced | 204.32 MB | 50.58 MB | 75.2% |

Small metric tables and the t-SNE sample remain CSV files.

The deployment flow is:

```text
GitHub Code
     │
     ▼
Streamlit Community Cloud
     │
     ├────────► Application UI
     │
     ▼
Hugging Face Dataset Repository
     │
     ▼
Parquet / CSV Analytical Data
```

## 📁 Repository Structure

```text
PatrolIQ/
│
├── app.py
├── data_loader.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
├── pages/
│   ├── 1_Crime_Dashboard.py
│   ├── 2_Temporal_Analysis.py
│   ├── 3_Crime_Map.py
│   ├── 4_Clustering_Results.py
│   ├── 5_Model_Performance.py
│   └── 6_Dimensionality_Reduction.py
│
└── README.md
```

The analytical development workflow also contains notebooks for:

```text
01_data_cleaning.ipynb
02_eda.ipynb
03_clustering.ipynb
04_dimensionality_reduction.ipynb
05_mlflow_tracking.ipynb
```

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/athiyamanomen-web/PatrolIQ-Smart-Safety-Analytics.git
cd PatrolIQ-Smart-Safety-Analytics
```

Create and activate a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application retrieves its deployment datasets from the public Hugging Face dataset repository.


## 🐳 Docker Deployment

PatrolIQ also includes containerization support for reproducible local deployment.

### Docker

Build the image from the repository root:

```bash
docker build -t patroliq .
```

Run the container:

```bash
docker run -p 8501:8501 patroliq
```

Then open the application at `http://localhost:8501`.

### Docker Compose

Alternatively, start PatrolIQ with Docker Compose:

```bash
docker compose up --build
```

Stop the container with:

```bash
docker compose down
```

The container exposes Streamlit on port **8501** and includes a health check for the Streamlit service. Deployment datasets continue to be retrieved from the public Hugging Face dataset repository, so the large analytical data files do not need to be bundled into the Docker image.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- MLflow
- PyArrow / Parquet
- Git & GitHub
- Hugging Face Datasets
- Streamlit Community Cloud
- Docker & Docker Compose

## ⚠️ Limitations

- The implementation analyzes the 500,000 most recent cleaned records from the available 847,276-record source rather than processing the complete multi-million-record historical Chicago crime archive.
- The analytical sample contains 31 crime types rather than the 33 referenced in the project brief.
- Crime counts are not population-adjusted risk estimates.
- Geographic clusters represent concentrations within the analyzed records and should not be interpreted as predictions of individual criminal activity.
- Hierarchical clustering metrics are based on a 10,000-record sample and are not directly comparable with full-dataset K-Means metrics.
- t-SNE is used for visualization on a 20,000-record sample.
- The severity score is a project-defined analytical heuristic.
- Temporal comparisons may be affected by unequal date coverage.

## 🎯 Conclusion

PatrolIQ demonstrates a complete data-science workflow from raw crime records to a deployed interactive analytics platform. The project combines large-scale data preprocessing, exploratory analysis, geographic and temporal clustering, dimensionality reduction, experiment tracking, and cloud deployment.

Geographic K-Means with **K=9** was selected as the deployment hotspot model because it produced a **0.5012 silhouette score on all 500,000 analytical records** while remaining scalable and interpretable. PCA reduced the engineered feature space from **22 dimensions to 3 principal components while retaining 71.77% of total variance**.

The completed Streamlit application makes these results explorable through interactive crime, temporal, geographic, clustering, model-performance, PCA, and t-SNE views.
