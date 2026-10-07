# ============================================================
# PatrolIQ - Geographic Crime Hotspots
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_clustered_data

from scipy.spatial import ConvexHull


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Geographic Hotspots",
    page_icon="🗺️",
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

    st.error(
        "Unable to load the PatrolIQ geographic dataset."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 4. Page Header
# ============================================================

st.title("🗺️ Geographic Crime Hotspots")

st.markdown(
    """
    Explore the geographic distribution of recorded Chicago crimes,
    crime-density hotspots, and the nine geographic zones identified
    by the PatrolIQ K-Means clustering model.
    """
)


# ============================================================
# 5. Interactive Geographic Filters
# ============================================================

st.sidebar.header("Geographic Filters")

st.sidebar.caption(
    "Leave a filter empty to include all values. "
    "Select one or more values to focus the entire geographic dashboard."
)


# ------------------------------------------------------------
# Reset All Filters
# ------------------------------------------------------------

filter_keys = [
    "geo_year",
    "geo_crime_type",
    "geo_district",
    "geo_zone",
    "geo_community",
    "geo_arrest",
    "geo_domestic",
]

if st.sidebar.button(
    "🔄 Reset All Filters",
    width="stretch"
):

    for key in filter_keys:

        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


# ------------------------------------------------------------
# Available Filter Values
# ------------------------------------------------------------

years = sorted(
    df["Year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

crime_types = sorted(
    df["Primary Type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

districts = sorted(
    df["District"]
    .dropna()
    .unique()
    .tolist()
)

clusters = sorted(
    df["KMeans_Cluster"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

community_areas = sorted(
    df["Community Area"]
    .dropna()
    .unique()
    .tolist()
)


# ------------------------------------------------------------
# Interactive Filter Controls
#
# Empty selection = ALL values
# ------------------------------------------------------------

selected_years = st.sidebar.multiselect(
    "Year",
    options=years,
    default=[],
    placeholder="All Years",
    key="geo_year"
)

selected_types = st.sidebar.multiselect(
    "Crime Type",
    options=crime_types,
    default=[],
    placeholder="All Crime Types",
    key="geo_crime_type"
)

selected_districts = st.sidebar.multiselect(
    "District",
    options=districts,
    default=[],
    placeholder="All Districts",
    key="geo_district"
)

selected_clusters = st.sidebar.multiselect(
    "Geographic Zone",
    options=clusters,
    default=[],
    placeholder="All Geographic Zones",
    key="geo_zone"
)

selected_communities = st.sidebar.multiselect(
    "Community Area",
    options=community_areas,
    default=[],
    placeholder="All Community Areas",
    key="geo_community"
)

selected_arrest = st.sidebar.multiselect(
    "Arrest Status",
    options=[
        "Arrested",
        "Not Arrested"
    ],
    default=[],
    placeholder="All Arrest Statuses",
    key="geo_arrest"
)

selected_domestic = st.sidebar.multiselect(
    "Domestic Status",
    options=[
        "Domestic",
        "Non-Domestic"
    ],
    default=[],
    placeholder="All Domestic Statuses",
    key="geo_domestic"
)


# ============================================================
# 6. Apply Filters
# ============================================================

filtered_df = df


# ------------------------------------------------------------
# Year
# ------------------------------------------------------------

if selected_years:

    filtered_df = filtered_df[
        filtered_df["Year"]
        .astype(int)
        .isin(selected_years)
    ]


# ------------------------------------------------------------
# Crime Type
# ------------------------------------------------------------

if selected_types:

    filtered_df = filtered_df[
        filtered_df["Primary Type"]
        .astype(str)
        .isin(selected_types)
    ]


# ------------------------------------------------------------
# District
# ------------------------------------------------------------

if selected_districts:

    filtered_df = filtered_df[
        filtered_df["District"]
        .isin(selected_districts)
    ]


# ------------------------------------------------------------
# Geographic Zone
# ------------------------------------------------------------

if selected_clusters:

    filtered_df = filtered_df[
        filtered_df["KMeans_Cluster"]
        .astype(int)
        .isin(selected_clusters)
    ]


# ------------------------------------------------------------
# Community Area
# ------------------------------------------------------------

if selected_communities:

    filtered_df = filtered_df[
        filtered_df["Community Area"]
        .isin(selected_communities)
    ]


# ------------------------------------------------------------
# Arrest Status
# ------------------------------------------------------------

if selected_arrest:

    arrest_values = []

    if "Arrested" in selected_arrest:
        arrest_values.append(True)

    if "Not Arrested" in selected_arrest:
        arrest_values.append(False)

    filtered_df = filtered_df[
        filtered_df["Arrest"].isin(arrest_values)
    ]


# ------------------------------------------------------------
# Domestic Status
# ------------------------------------------------------------

if selected_domestic:

    domestic_values = []

    if "Domestic" in selected_domestic:
        domestic_values.append(True)

    if "Non-Domestic" in selected_domestic:
        domestic_values.append(False)

    filtered_df = filtered_df[
        filtered_df["Domestic"].isin(domestic_values)
    ]


# ============================================================
# 7. Active Filter Summary
# ============================================================

active_filters = []


if selected_years:

    active_filters.append(
        "Year: "
        + ", ".join(
            map(str, selected_years)
        )
    )


if selected_types:

    active_filters.append(
        "Crime Type: "
        + ", ".join(selected_types)
    )


if selected_districts:

    active_filters.append(
        "District: "
        + ", ".join(
            map(str, selected_districts)
        )
    )


if selected_clusters:

    active_filters.append(
        "Geographic Zone: "
        + ", ".join(
            map(str, selected_clusters)
        )
    )


if selected_communities:

    active_filters.append(
        "Community Area: "
        + ", ".join(
            map(str, selected_communities)
        )
    )


if selected_arrest:

    active_filters.append(
        "Arrest: "
        + ", ".join(selected_arrest)
    )


if selected_domestic:

    active_filters.append(
        "Domestic: "
        + ", ".join(selected_domestic)
    )


st.sidebar.divider()


if active_filters:

    st.sidebar.success(
        f"Showing {len(filtered_df):,} "
        f"of {len(df):,} records"
    )

    with st.sidebar.expander(
        f"Active Filters ({len(active_filters)})"
    ):

        for item in active_filters:
            st.write(f"• {item}")

else:

    st.sidebar.info(
        f"Showing all {len(df):,} records"
    )


# ------------------------------------------------------------
# Handle Empty Result
# ------------------------------------------------------------

if filtered_df.empty:

    st.warning(
        "No crime records match the selected geographic filters. "
        "Change a filter or click Reset All Filters."
    )

    st.stop()


# ============================================================
# 8. Summary Metrics
# ============================================================

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Mapped Crimes",
    f"{len(filtered_df):,}"
)


col2.metric(
    "Geographic Zones",
    filtered_df["KMeans_Cluster"].nunique()
)


col3.metric(
    "Districts",
    filtered_df["District"].nunique()
)


col4.metric(
    "Crime Types",
    filtered_df["Primary Type"].nunique()
)


st.caption(
    "Geographic zones represent clustering of recorded crime "
    "locations and administrative geography. They are not "
    "population-adjusted measures of personal safety."
)


st.divider()


# ============================================================
# 9. Visualization Sample Size
# ============================================================

max_map_sample = min(
    20000,
    len(filtered_df)
)


if max_map_sample >= 5000:

    default_map_sample = min(
        10000,
        max_map_sample
    )

    map_sample_size = st.slider(
        "Interactive Map Sample Size",
        min_value=5000,
        max_value=max_map_sample,
        value=default_map_sample,
        step=5000
    )

else:

    map_sample_size = len(filtered_df)

    st.caption(
        f"Interactive map is displaying all "
        f"{map_sample_size:,} filtered records."
    )


map_df = filtered_df.sample(
    n=min(
        map_sample_size,
        len(filtered_df)
    ),
    random_state=42
).copy()


map_df["Geographic Zone"] = (
    map_df["KMeans_Cluster"]
    .astype(str)
)


# ============================================================
# 10. Geographic Crime Density Heatmap
# ============================================================

st.subheader("Crime Density Heatmap")

heatmap_sample = filtered_df.sample(
    n=min(15000, len(filtered_df)),
    random_state=42
).copy()

# Aggregate nearby crime locations into a lightweight density surface.
# Scattermap is also used by the working boundary visualization below.
lat_bins = np.linspace(
    heatmap_sample["Latitude"].min(),
    heatmap_sample["Latitude"].max(),
    46
)
lon_bins = np.linspace(
    heatmap_sample["Longitude"].min(),
    heatmap_sample["Longitude"].max(),
    46
)

heatmap_sample["lat_bin"] = pd.cut(
    heatmap_sample["Latitude"],
    bins=lat_bins,
    labels=False,
    include_lowest=True
)
heatmap_sample["lon_bin"] = pd.cut(
    heatmap_sample["Longitude"],
    bins=lon_bins,
    labels=False,
    include_lowest=True
)

density_df = (
    heatmap_sample
    .dropna(subset=["lat_bin", "lon_bin"])
    .groupby(["lat_bin", "lon_bin"], observed=True)
    .agg(
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean"),
        Crime_Count=("Latitude", "size")
    )
    .reset_index()
)

max_density = max(int(density_df["Crime_Count"].max()), 1)
density_df["Marker_Size"] = (
    5 + 28 * np.sqrt(density_df["Crime_Count"] / max_density)
)

fig_heatmap = go.Figure()

fig_heatmap.add_trace(
    go.Scattermap(
        lat=density_df["Latitude"],
        lon=density_df["Longitude"],
        mode="markers",
        marker={
            "size": density_df["Marker_Size"],
            "color": density_df["Crime_Count"],
            "colorscale": "Hot",
            "showscale": True,
            "opacity": 0.72,
            "colorbar": {"title": {"text": "Crime<br>Count"}}
        },
        text="Recorded crimes: " + density_df["Crime_Count"].astype(str),
        hovertemplate=(
            "%{text}<br>"
            "Latitude: %{lat:.4f}<br>"
            "Longitude: %{lon:.4f}"
            "<extra></extra>"
        ),
        name="Crime density"
    )
)

fig_heatmap.update_layout(
    map={
        "style": "carto-darkmatter",
        "zoom": 9,
        "center": {
            "lat": heatmap_sample["Latitude"].mean(),
            "lon": heatmap_sample["Longitude"].mean()
        }
    },
    height=650,
    title="Chicago Recorded Crime Density",
    margin={"r": 0, "t": 45, "l": 0, "b": 0}
)

st.plotly_chart(fig_heatmap, width="stretch")

st.caption(
    "Larger and brighter markers represent higher concentrations of "
    "recorded crime locations in the selected data."
)


# ============================================================
# 11. K-Means Geographic Hotspot Zones
# ============================================================

st.subheader("K-Means Geographic Hotspot Zones")

fig_clusters = go.Figure()

zone_palette = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF"
]

for zone_position, zone in enumerate(
    sorted(map_df["KMeans_Cluster"].dropna().astype(int).unique())
):
    zone_points = map_df[
        map_df["KMeans_Cluster"].astype(int) == zone
    ].copy()

    customdata = np.column_stack([
        zone_points["Primary Type"].astype(str),
        zone_points["Description"].astype(str),
        zone_points["District"].astype(str),
        zone_points["Beat"].astype(str),
        zone_points["Community Area"].astype(str),
        zone_points["Arrest"].astype(str),
        zone_points["Domestic"].astype(str)
    ])

    fig_clusters.add_trace(
        go.Scattermap(
            lat=zone_points["Latitude"],
            lon=zone_points["Longitude"],
            mode="markers",
            marker={
                "size": 5,
                "color": zone_palette[zone_position % len(zone_palette)],
                "opacity": 0.55
            },
            customdata=customdata,
            hovertemplate=(
                f"Geographic Zone {zone}<br>"
                "Crime: %{customdata[0]}<br>"
                "Description: %{customdata[1]}<br>"
                "District: %{customdata[2]}<br>"
                "Beat: %{customdata[3]}<br>"
                "Community Area: %{customdata[4]}<br>"
                "Arrest: %{customdata[5]}<br>"
                "Domestic: %{customdata[6]}"
                "<extra></extra>"
            ),
            name=f"Zone {zone}"
        )
    )

fig_clusters.update_layout(
    map={
        "style": "carto-darkmatter",
        "zoom": 9,
        "center": {
            "lat": map_df["Latitude"].mean(),
            "lon": map_df["Longitude"].mean()
        }
    },
    height=700,
    title="PatrolIQ K-Means Geographic Zones — K = 9",
    margin={"r": 0, "t": 45, "l": 0, "b": 0},
    legend={"title": {"text": "Geographic Zone"}}
)

st.plotly_chart(fig_clusters, width="stretch")


# ============================================================
# 12. Approximate Geographic Cluster Boundaries
# ============================================================

st.subheader(
    "Geographic Zone Boundaries"
)


st.markdown(
    """
    The polygons below provide an approximate visual envelope
    around the recorded crime locations assigned to each K-Means
    geographic zone.
    """
)


boundary_sample = filtered_df.sample(
    n=min(
        20000,
        len(filtered_df)
    ),
    random_state=42
).copy()


fig_boundary = go.Figure()


# ------------------------------------------------------------
# Add Sampled Crime Locations
# ------------------------------------------------------------

point_sample = boundary_sample.sample(
    n=min(
        5000,
        len(boundary_sample)
    ),
    random_state=42
)


fig_boundary.add_trace(
    go.Scattermap(
        lat=point_sample["Latitude"],
        lon=point_sample["Longitude"],
        mode="markers",
        marker={
            "size": 3,
            "opacity": 0.25
        },
        text=(
            "Zone "
            + point_sample[
                "KMeans_Cluster"
            ].astype(str)
        ),
        hoverinfo="text",
        name="Crime locations"
    )
)


# ------------------------------------------------------------
# Create Convex-Hull Boundary for Each Selected Zone
# ------------------------------------------------------------

boundary_centers = []


for cluster in sorted(
    boundary_sample[
        "KMeans_Cluster"
    ].unique()
):

    cluster_data = boundary_sample[
        boundary_sample[
            "KMeans_Cluster"
        ] == cluster
    ][
        [
            "Longitude",
            "Latitude"
        ]
    ].dropna()


    if len(cluster_data) < 3:
        continue


    # --------------------------------------------------------
    # Trim extreme geographic outliers before calculating
    # the visual boundary.
    # --------------------------------------------------------

    lon_low = (
        cluster_data[
            "Longitude"
        ].quantile(0.02)
    )

    lon_high = (
        cluster_data[
            "Longitude"
        ].quantile(0.98)
    )

    lat_low = (
        cluster_data[
            "Latitude"
        ].quantile(0.02)
    )

    lat_high = (
        cluster_data[
            "Latitude"
        ].quantile(0.98)
    )


    cluster_trimmed = cluster_data[
        cluster_data[
            "Longitude"
        ].between(
            lon_low,
            lon_high
        )
        &
        cluster_data[
            "Latitude"
        ].between(
            lat_low,
            lat_high
        )
    ].copy()


    if len(cluster_trimmed) < 3:
        continue


    points = cluster_trimmed[
        [
            "Longitude",
            "Latitude"
        ]
    ].to_numpy()


    try:

        hull = ConvexHull(points)

        hull_points = points[
            hull.vertices
        ]


        # Close polygon
        hull_points = np.vstack(
            [
                hull_points,
                hull_points[0]
            ]
        )


        fig_boundary.add_trace(
            go.Scattermap(
                lon=hull_points[:, 0],
                lat=hull_points[:, 1],
                mode="lines",
                line={
                    "width": 3
                },
                name=f"Zone {cluster}",
                hovertemplate=(
                    f"Geographic Zone {cluster}"
                    "<extra></extra>"
                )
            )
        )


        boundary_centers.append(
            {
                "Zone": cluster,

                "Latitude":
                    cluster_trimmed[
                        "Latitude"
                    ].mean(),

                "Longitude":
                    cluster_trimmed[
                        "Longitude"
                    ].mean()
            }
        )


    except Exception:
        continue


# ------------------------------------------------------------
# Add Geographic Zone Center Labels
# ------------------------------------------------------------

if boundary_centers:

    centers_df = pd.DataFrame(
        boundary_centers
    )


    fig_boundary.add_trace(
        go.Scattermap(
            lat=centers_df["Latitude"],
            lon=centers_df["Longitude"],
            mode="markers+text",
            marker={
                "size": 12
            },
            text=(
                "Zone "
                + centers_df[
                    "Zone"
                ].astype(str)
            ),
            textposition="top center",
            hovertemplate=(
                "Geographic %{text}"
                "<extra></extra>"
            ),
            name="Zone centers"
        )
    )


fig_boundary.update_layout(
    map={
        "style": "carto-darkmatter",
        "zoom": 9,
        "center": {
            "lat":
                filtered_df[
                    "Latitude"
                ].mean(),

            "lon":
                filtered_df[
                    "Longitude"
                ].mean()
        }
    },

    height=720,

    margin={
        "r": 0,
        "t": 30,
        "l": 0,
        "b": 0
    },

    legend={
        "title":
            "Cluster Boundaries"
    }
)


st.plotly_chart(
    fig_boundary,
    width="stretch"
)


st.caption(
    "Boundary polygons are visualization envelopes derived from "
    "the spatial extent of records assigned to each K-Means zone. "
    "They are not official Chicago administrative boundaries."
)


# ============================================================
# 13. Geographic Zone Volume
# ============================================================

st.subheader(
    "Geographic Zone Profiles"
)


cluster_counts = (
    filtered_df[
        "KMeans_Cluster"
    ]
    .value_counts()
    .sort_index()
    .reset_index()
)


cluster_counts.columns = [
    "Geographic Zone",
    "Crime Count"
]


cluster_counts[
    "Geographic Zone"
] = (
    cluster_counts[
        "Geographic Zone"
    ].astype(str)
)


fig_cluster_counts = px.bar(
    cluster_counts,
    x="Geographic Zone",
    y="Crime Count",
    color="Geographic Zone",
    title=(
        "Recorded Crime Volume "
        "by Geographic Zone"
    )
)


st.plotly_chart(
    fig_cluster_counts,
    width="stretch"
)


# ============================================================
# 14. Geographic Zone Profile Table
# ============================================================

profile_rows = []


for cluster in sorted(
    filtered_df[
        "KMeans_Cluster"
    ].unique()
):

    cluster_df = filtered_df[
        filtered_df[
            "KMeans_Cluster"
        ] == cluster
    ]


    top_crime = (
        cluster_df[
            "Primary Type"
        ].mode()
    )

    top_district = (
        cluster_df[
            "District"
        ].mode()
    )


    profile_rows.append(
        {
            "Geographic Zone":
                int(cluster),

            "Crime Records":
                len(cluster_df),

            "Share (%)":
                round(
                    len(cluster_df)
                    / len(filtered_df)
                    * 100,
                    2
                ),

            "Top Crime Type":
                (
                    top_crime.iloc[0]
                    if not top_crime.empty
                    else "N/A"
                ),

            "Top District":
                (
                    top_district.iloc[0]
                    if not top_district.empty
                    else "N/A"
                ),

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
    )


profile_df = pd.DataFrame(
    profile_rows
)


st.dataframe(
    profile_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# 15. Geographic Zone Detail Explorer
# ============================================================

st.subheader(
    "Geographic Zone Detail Explorer"
)


available_zones = sorted(
    filtered_df[
        "KMeans_Cluster"
    ]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)


if available_zones:

    detail_zone = st.selectbox(
        "Select a Geographic Zone to inspect",
        options=available_zones
    )


    zone_df = filtered_df[
        filtered_df[
            "KMeans_Cluster"
        ].astype(int)
        == detail_zone
    ].copy()


    z1, z2, z3, z4 = st.columns(4)


    z1.metric(
        "Zone Records",
        f"{len(zone_df):,}"
    )


    z2.metric(
        "Crime Types",
        zone_df[
            "Primary Type"
        ].nunique()
    )


    z3.metric(
        "Arrest Rate",
        f"{zone_df['Arrest'].mean() * 100:.2f}%"
    )


    z4.metric(
        "Domestic Rate",
        f"{zone_df['Domestic'].mean() * 100:.2f}%"
    )


    zone_crimes = (
        zone_df[
            "Primary Type"
        ]
        .value_counts()
        .head(10)
        .reset_index()
    )


    zone_crimes.columns = [
        "Crime Type",
        "Crime Count"
    ]


    fig_zone_crimes = px.bar(
        zone_crimes,
        x="Crime Count",
        y="Crime Type",
        orientation="h",
        title=(
            f"Top Crime Types "
            f"in Geographic Zone {detail_zone}"
        )
    )


    fig_zone_crimes.update_layout(
        yaxis={
            "categoryorder":
                "total ascending"
        }
    )


    st.plotly_chart(
        fig_zone_crimes,
        width="stretch"
    )


# ============================================================
# 16. Selected Geographic Records
# ============================================================

st.subheader(
    "Selected Geographic Records"
)


display_columns = [
    "Date",
    "Primary Type",
    "Description",
    "Location Description",
    "District",
    "Ward",
    "Beat",
    "Community Area",
    "Latitude",
    "Longitude",
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
    filtered_df[
        available_columns
    ].head(500),
    width="stretch",
    hide_index=True
)


# ============================================================
# 17. Download Filtered Geographic Data
# ============================================================

st.subheader(
    "Export Selected Geographic Data"
)


st.caption(
    "To keep the cloud app responsive, the CSV export is prepared only when requested."
)

if st.button("Prepare Geographic CSV Export"):
    export_csv = (
        filtered_df[available_columns]
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Filtered Geographic Data",
        data=export_csv,
        file_name="patroliq_geographic_filtered.csv",
        mime="text/csv",
        width="content"
    )


# ============================================================
# 18. Interpretation
# ============================================================

st.info(
    """
    PatrolIQ uses K-Means with K=9 as the geographic deployment
    clustering model. The model achieved a Silhouette Score of
    0.5012 and a Davies-Bouldin Index of 0.7153 on the complete
    500,000-record analytical dataset.

    The interactive filters allow geographic crime patterns to be
    explored by year, crime type, district, geographic zone,
    community area, arrest status, and domestic status.

    Hotspot volume represents the concentration of recorded crimes
    in this dataset. It should not be interpreted as
    population-adjusted crime risk or as a prediction of future crime.
    """
)