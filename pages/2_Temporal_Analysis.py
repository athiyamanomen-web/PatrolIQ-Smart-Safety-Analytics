# ============================================================
# PatrolIQ - Temporal Pattern Analysis
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Temporal Analysis",
    page_icon="⏰",
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

    st.error("Unable to load the PatrolIQ dataset.")
    st.exception(error)
    st.stop()


# ============================================================
# 4. Page Header
# ============================================================

st.title("⏰ Temporal Crime Pattern Analysis")

st.markdown(
    """
    Explore when crimes are recorded across Chicago and examine
    the four temporal behavior patterns identified by Temporal K-Means.
    """
)


# ============================================================
# 5. Interactive Temporal Filters
# ============================================================

st.sidebar.header("Temporal Filters")
st.sidebar.caption(
    "Leave a filter empty to include all values. "
    "Select one or more values to update the entire temporal analysis."
)

temporal_filter_keys = [
    "temporal_year", "temporal_month", "temporal_crime_type",
    "temporal_pattern", "temporal_day", "temporal_weekend",
]

if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    for key in temporal_filter_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

years = sorted(df["Year"].dropna().unique().tolist())
months = sorted(df["Month"].dropna().unique().tolist())
crime_types = sorted(df["Primary Type"].dropna().unique().tolist())
temporal_clusters = sorted(
    df["Temporal_Cluster"].dropna().unique().tolist()
)

day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
available_days = [
    day for day in day_order
    if day in df["Day_of_Week"].dropna().unique()
]

selected_years = st.sidebar.multiselect(
    "Year", years, default=[], placeholder="All Years", key="temporal_year"
)
selected_months = st.sidebar.multiselect(
    "Month", months, default=[], placeholder="All Months", key="temporal_month"
)
selected_types = st.sidebar.multiselect(
    "Crime Type", crime_types, default=[], placeholder="All Crime Types",
    key="temporal_crime_type"
)
selected_clusters = st.sidebar.multiselect(
    "Temporal Pattern", temporal_clusters, default=[],
    placeholder="All Temporal Patterns", key="temporal_pattern"
)
selected_days = st.sidebar.multiselect(
    "Day of Week", available_days, default=[], placeholder="All Days",
    key="temporal_day"
)
selected_weekend = st.sidebar.selectbox(
    "Day Category", ["All", "Weekday", "Weekend"], key="temporal_weekend"
)

filtered_df = df.copy()

if selected_years:
    filtered_df = filtered_df[filtered_df["Year"].isin(selected_years)]
if selected_months:
    filtered_df = filtered_df[filtered_df["Month"].isin(selected_months)]
if selected_types:
    filtered_df = filtered_df[filtered_df["Primary Type"].isin(selected_types)]
if selected_clusters:
    filtered_df = filtered_df[
        filtered_df["Temporal_Cluster"].isin(selected_clusters)
    ]
if selected_days:
    filtered_df = filtered_df[filtered_df["Day_of_Week"].isin(selected_days)]

if selected_weekend == "Weekday":
    filtered_df = filtered_df[filtered_df["Is_Weekend"] == 0]
elif selected_weekend == "Weekend":
    filtered_df = filtered_df[filtered_df["Is_Weekend"] == 1]

active_filters = []
if selected_years:
    active_filters.append("Year: " + ", ".join(map(str, selected_years)))
if selected_months:
    active_filters.append("Month: " + ", ".join(map(str, selected_months)))
if selected_types:
    active_filters.append("Crime Type: " + ", ".join(map(str, selected_types)))
if selected_clusters:
    active_filters.append(
        "Temporal Pattern: " + ", ".join(map(str, selected_clusters))
    )
if selected_days:
    active_filters.append("Day: " + ", ".join(selected_days))
if selected_weekend != "All":
    active_filters.append("Day Category: " + selected_weekend)

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
        "No crime records match the selected temporal filters. "
        "Change a filter or click Reset All Filters."
    )
    st.stop()


# ============================================================
# 6. Summary Metrics
# ============================================================

peak_hour = int(
    filtered_df["Hour"]
    .value_counts()
    .idxmax()
)

peak_day = (
    filtered_df["Day_of_Week"]
    .value_counts()
    .idxmax()
)

peak_month_number = int(
    filtered_df["Month"]
    .value_counts()
    .idxmax()
)

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

peak_month = month_names.get(
    peak_month_number,
    str(peak_month_number)
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Filtered Crimes",
    f"{len(filtered_df):,}"
)


col2.metric(
    "Peak Recorded Hour",
    f"{peak_hour:02d}:00"
)


col3.metric(
    "Peak Recorded Day",
    peak_day
)


col4.metric(
    "Peak Recorded Month",
    peak_month
)


st.caption(
    "Peak values describe the highest recorded counts in the "
    "selected data and should not be interpreted as population-adjusted risk."
)


st.divider()


# ============================================================
# 7. Crimes by Hour
# ============================================================

hourly = (
    filtered_df["Hour"]
    .value_counts()
    .sort_index()
    .reset_index()
)

hourly.columns = [
    "Hour",
    "Crime Count"
]


fig_hour = px.line(
    hourly,
    x="Hour",
    y="Crime Count",
    markers=True,
    title="Recorded Crimes by Hour"
)

fig_hour.update_xaxes(
    dtick=1
)


st.plotly_chart(
    fig_hour,
    use_container_width=True
)


# ============================================================
# 8. Day and Month Distribution
# ============================================================

col1, col2 = st.columns(2)


day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


with col1:

    daily = (
        filtered_df["Day_of_Week"]
        .value_counts()
        .reindex(day_order, fill_value=0)
        .reset_index()
    )

    daily.columns = [
        "Day",
        "Crime Count"
    ]

    fig_day = px.bar(
        daily,
        x="Day",
        y="Crime Count",
        title="Recorded Crimes by Day of Week"
    )

    st.plotly_chart(
        fig_day,
        use_container_width=True
    )


with col2:

    monthly = (
        filtered_df["Month"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    monthly.columns = [
        "Month",
        "Crime Count"
    ]

    monthly["Month Name"] = (
        monthly["Month"]
        .map({
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec"
        })
    )

    fig_month = px.bar(
        monthly,
        x="Month Name",
        y="Crime Count",
        title="Recorded Crimes by Month"
    )

    st.plotly_chart(
        fig_month,
        use_container_width=True
    )


# ============================================================
# 9. Hour × Day-of-Week Heatmap
# ============================================================

st.subheader(
    "Hourly Crime Heatmap"
)


heatmap_data = pd.crosstab(
    filtered_df["Day_of_Week"],
    filtered_df["Hour"]
)


heatmap_data = heatmap_data.reindex(
    day_order,
    fill_value=0
)


fig_heatmap = px.imshow(
    heatmap_data,
    labels={
        "x": "Hour",
        "y": "Day of Week",
        "color": "Crime Count"
    },
    aspect="auto",
    title="Recorded Crime Count — Hour × Day of Week"
)


st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)


# ============================================================
# 10. Weekday vs Weekend
# ============================================================

col1, col2 = st.columns(2)


with col1:

    weekend_counts = (
        filtered_df["Is_Weekend"]
        .map({
            0: "Weekday",
            1: "Weekend"
        })
        .value_counts()
        .reset_index()
    )

    weekend_counts.columns = [
        "Period",
        "Crime Count"
    ]

    fig_weekend = px.pie(
        weekend_counts,
        names="Period",
        values="Crime Count",
        title="Weekday vs Weekend Recorded Crime"
    )

    st.plotly_chart(
        fig_weekend,
        use_container_width=True
    )


# ============================================================
# 11. Seasonal Distribution
# ============================================================

with col2:

    if "Season" in filtered_df.columns:

        season_order = [
            "Winter",
            "Spring",
            "Summer",
            "Fall"
        ]

        seasonal = (
            filtered_df["Season"]
            .value_counts()
            .reindex(
                season_order,
                fill_value=0
            )
            .reset_index()
        )

        seasonal.columns = [
            "Season",
            "Crime Count"
        ]

        fig_season = px.bar(
            seasonal,
            x="Season",
            y="Crime Count",
            title="Recorded Crime by Season"
        )

        st.plotly_chart(
            fig_season,
            use_container_width=True
        )

    else:

        st.info(
            "Season column is not available in the dataset."
        )


# ============================================================
# 12. Temporal K-Means Pattern Distribution
# ============================================================

st.subheader(
    "Temporal K-Means Patterns"
)


cluster_counts = (
    filtered_df["Temporal_Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "Temporal Pattern",
    "Crime Count"
]


cluster_counts["Temporal Pattern"] = (
    cluster_counts["Temporal Pattern"]
    .astype(str)
)


fig_cluster = px.bar(
    cluster_counts,
    x="Temporal Pattern",
    y="Crime Count",
    color="Temporal Pattern",
    title="Crime Records by Temporal Pattern"
)


st.plotly_chart(
    fig_cluster,
    use_container_width=True
)


# ============================================================
# 13. Temporal Cluster Profiles
# ============================================================

st.subheader(
    "Temporal Pattern Profiles"
)


profile_agg = {
    "Hour": "mean",
    "Month": "mean"
}


if "Crime_Severity_Score" in filtered_df.columns:
    profile_agg["Crime_Severity_Score"] = "mean"


if "Is_Weekend" in filtered_df.columns:
    profile_agg["Is_Weekend"] = "mean"


temporal_profiles = (
    filtered_df
    .groupby("Temporal_Cluster")
    .agg(profile_agg)
    .round(2)
    .reset_index()
)


temporal_profiles = temporal_profiles.rename(
    columns={
        "Temporal_Cluster": "Temporal Pattern",
        "Hour": "Average Hour",
        "Month": "Average Month",
        "Crime_Severity_Score": "Average Severity",
        "Is_Weekend": "Weekend Share"
    }
)


if "Weekend Share" in temporal_profiles.columns:

    temporal_profiles["Weekend Share"] = (
        temporal_profiles["Weekend Share"] * 100
    ).round(2)

    temporal_profiles = temporal_profiles.rename(
        columns={
            "Weekend Share": "Weekend Share (%)"
        }
    )


st.dataframe(
    temporal_profiles,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 14. Crime Type × Temporal Pattern
# ============================================================

st.subheader(
    "Crime Types Across Temporal Patterns"
)


top_types = (
    filtered_df["Primary Type"]
    .value_counts()
    .head(10)
    .index
)


crime_temporal = (
    filtered_df[
        filtered_df["Primary Type"].isin(top_types)
    ]
    .groupby(
        [
            "Primary Type",
            "Temporal_Cluster"
        ]
    )
    .size()
    .reset_index(
        name="Crime Count"
    )
)


crime_temporal["Temporal Pattern"] = (
    crime_temporal["Temporal_Cluster"]
    .astype(str)
)


fig_crime_temporal = px.bar(
    crime_temporal,
    x="Primary Type",
    y="Crime Count",
    color="Temporal Pattern",
    barmode="group",
    title="Top Crime Types by Temporal Pattern"
)


st.plotly_chart(
    fig_crime_temporal,
    use_container_width=True
)


# ============================================================
# 15. Interpretation
# ============================================================

st.info(
    """
    Temporal K-Means groups records with similar combinations of
    hour, day, month, severity, and weekend behavior. These clusters
    describe recurring temporal crime patterns rather than predictions
    of when an individual crime will occur.
    """
)