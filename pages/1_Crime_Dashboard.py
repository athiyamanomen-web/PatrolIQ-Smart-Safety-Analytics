# ============================================================
# PatrolIQ - Crime Analysis Dashboard
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_clustered_data


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Crime Dashboard",
    page_icon="📊",
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

try:

    df = load_clustered_data()

except Exception as error:

    st.error("Unable to load the PatrolIQ dataset.")
    st.exception(error)
    st.stop()


# ============================================================
# 4. Page Header
# ============================================================

st.title("📊 Crime Analysis Dashboard")

st.markdown(
    """
    Explore reported Chicago crime patterns using interactive
    filters for year, month, crime type, and police district.
    """
)


# ============================================================
# 5. Interactive Dashboard Filters
# ============================================================

st.sidebar.header("Dashboard Filters")
st.sidebar.caption(
    "Leave a filter empty to include all values. "
    "Select one or more values to update the entire dashboard."
)

dashboard_filter_keys = [
    "dashboard_year", "dashboard_month", "dashboard_crime",
    "dashboard_district", "dashboard_ward",
    "dashboard_arrest", "dashboard_domestic",
]

if st.sidebar.button("🔄 Reset All Filters", width="stretch"):
    for key in dashboard_filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

years = sorted(df["Year"].dropna().unique().tolist())
months = sorted(df["Month"].dropna().unique().tolist())
crime_types = sorted(df["Primary Type"].dropna().unique().tolist())
districts = sorted(df["District"].dropna().unique().tolist())
wards = sorted(df["Ward"].dropna().unique().tolist())

selected_year = st.sidebar.multiselect(
    "Year", years, default=[], placeholder="All Years", key="dashboard_year"
)
selected_month = st.sidebar.multiselect(
    "Month", months, default=[], placeholder="All Months", key="dashboard_month"
)
selected_type = st.sidebar.multiselect(
    "Crime Type", crime_types, default=[], placeholder="All Crime Types",
    key="dashboard_crime"
)
selected_district = st.sidebar.multiselect(
    "District", districts, default=[], placeholder="All Districts",
    key="dashboard_district"
)
selected_ward = st.sidebar.multiselect(
    "Ward", wards, default=[], placeholder="All Wards", key="dashboard_ward"
)
selected_arrest = st.sidebar.selectbox(
    "Arrest Status",
    ["All", "Arrest Recorded", "No Arrest Recorded"],
    key="dashboard_arrest"
)
selected_domestic = st.sidebar.selectbox(
    "Domestic Status",
    ["All", "Domestic", "Non-Domestic"],
    key="dashboard_domestic"
)

# ============================================================
# 6. Apply Filters
# ============================================================

filtered_df = df

if selected_year:
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_year)]
if selected_month:
    filtered_df = filtered_df[filtered_df["Month"].isin(selected_month)]
if selected_type:
    filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_type)]
if selected_district:
    filtered_df = filtered_df[filtered_df["District"].isin(selected_district)]
if selected_ward:
    filtered_df = filtered_df[filtered_df["Ward"].isin(selected_ward)]

if selected_arrest == "Arrest Recorded":
    filtered_df = filtered_df[filtered_df["Arrest"] == True]
elif selected_arrest == "No Arrest Recorded":
    filtered_df = filtered_df[filtered_df["Arrest"] == False]

if selected_domestic == "Domestic":
    filtered_df = filtered_df[filtered_df["Domestic"] == True]
elif selected_domestic == "Non-Domestic":
    filtered_df = filtered_df[filtered_df["Domestic"] == False]

active_filters = []
if selected_year:
    active_filters.append("Year: " + ", ".join(map(str, selected_year)))
if selected_month:
    active_filters.append("Month: " + ", ".join(map(str, selected_month)))
if selected_type:
    active_filters.append("Crime Type: " + ", ".join(map(str, selected_type)))
if selected_district:
    active_filters.append("District: " + ", ".join(map(str, selected_district)))
if selected_ward:
    active_filters.append("Ward: " + ", ".join(map(str, selected_ward)))
if selected_arrest != "All":
    active_filters.append("Arrest: " + selected_arrest)
if selected_domestic != "All":
    active_filters.append("Domestic: " + selected_domestic)

st.sidebar.divider()

if active_filters:
    st.sidebar.success(f"Showing {len(filtered_df):,} of {len(df):,} records")
    with st.sidebar.expander(f"Active Filters ({len(active_filters)})"):
        for item in active_filters:
            st.write(f"• {item}")
else:
    st.sidebar.info(f"Showing all {len(df):,} records")

if filtered_df.empty:
    st.warning(
        "No crime records match the selected filters. "
        "Change a filter or click Reset All Filters."
    )
    st.stop()


# ============================================================
# 7. Key Metrics
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)


col1.metric(
    "Filtered Crimes",
    f"{len(filtered_df):,}"
)


col2.metric(
    "Crime Types",
    filtered_df["Primary Type"].nunique()
)


col3.metric(
    "Districts",
    filtered_df["District"].nunique()
)


col4.metric(
    "Arrest Rate",
    f"{filtered_df['Arrest'].mean() * 100:.2f}%"
)


col5.metric(
    "Domestic Rate",
    f"{filtered_df['Domestic'].mean() * 100:.2f}%"
)


st.divider()


# ============================================================
# 8. Top Crime Types
# ============================================================

st.subheader("Crime Category Distribution")


top_crimes = (
    filtered_df["Primary Type"]
    .value_counts()
    .head(10)
    .reset_index()
)

top_crimes.columns = [
    "Crime Type",
    "Crime Count"
]


fig_crime = px.bar(
    top_crimes,
    x="Crime Count",
    y="Crime Type",
    orientation="h",
    title="Top 10 Crime Types"
)

fig_crime.update_layout(
    yaxis={
        "categoryorder": "total ascending"
    }
)

st.plotly_chart(
    fig_crime,
    width="stretch"
)


# ============================================================
# 9. District and Ward Distribution
# ============================================================

col1, col2 = st.columns(2)


with col1:

    district_counts = (
        filtered_df["District"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    district_counts.columns = [
        "District",
        "Crime Count"
    ]

    fig_district = px.bar(
        district_counts,
        x="District",
        y="Crime Count",
        title="Top Districts by Recorded Crime Count"
    )

    st.plotly_chart(
        fig_district,
        width="stretch"
    )


with col2:

    ward_counts = (
        filtered_df["Ward"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    ward_counts.columns = [
        "Ward",
        "Crime Count"
    ]

    fig_ward = px.bar(
        ward_counts,
        x="Ward",
        y="Crime Count",
        title="Top Wards by Recorded Crime Count"
    )

    st.plotly_chart(
        fig_ward,
        width="stretch"
    )


# ============================================================
# 10. Arrest and Domestic Distribution
# ============================================================

col1, col2 = st.columns(2)


with col1:

    arrest_counts = (
        filtered_df["Arrest"]
        .map({
            True: "Arrest Recorded",
            False: "No Arrest Recorded"
        })
        .value_counts()
        .reset_index()
    )

    arrest_counts.columns = [
        "Arrest Status",
        "Crime Count"
    ]

    fig_arrest = px.pie(
        arrest_counts,
        names="Arrest Status",
        values="Crime Count",
        title="Recorded Arrest Status"
    )

    st.plotly_chart(
        fig_arrest,
        width="stretch"
    )


with col2:

    domestic_counts = (
        filtered_df["Domestic"]
        .map({
            True: "Domestic",
            False: "Non-Domestic"
        })
        .value_counts()
        .reset_index()
    )

    domestic_counts.columns = [
        "Domestic Status",
        "Crime Count"
    ]

    fig_domestic = px.pie(
        domestic_counts,
        names="Domestic Status",
        values="Crime Count",
        title="Domestic vs Non-Domestic Crime"
    )

    st.plotly_chart(
        fig_domestic,
        width="stretch"
    )


# ============================================================
# 11. Crime Description Distribution
# ============================================================

st.subheader("Crime Description Analysis")


description_counts = (
    filtered_df["Description"]
    .value_counts()
    .head(15)
    .reset_index()
)

description_counts.columns = [
    "Description",
    "Crime Count"
]


fig_description = px.bar(
    description_counts,
    x="Crime Count",
    y="Description",
    orientation="h",
    title="Top 15 Recorded Crime Descriptions"
)

fig_description.update_layout(
    yaxis={
        "categoryorder": "total ascending"
    }
)


st.plotly_chart(
    fig_description,
    width="stretch"
)


# ============================================================
# 12. Filtered Records
# ============================================================

st.subheader("Filtered Crime Records")


display_columns = [
    "Date",
    "Primary Type",
    "Description",
    "Location Description",
    "District",
    "Ward",
    "Arrest",
    "Domestic",
    "KMeans_Cluster",
    "Temporal_Cluster"
]


available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]


st.dataframe(
    filtered_df[available_columns].head(500),
    width="stretch",
    hide_index=True
)


# ============================================================
# 13. Download Filtered Data
# ============================================================

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Filtered Crime Data",
    data=csv,
    file_name="patroliq_filtered_crime_data.csv",
    mime="text/csv"
)