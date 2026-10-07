from pathlib import Path

import pandas as pd
import streamlit as st

HF_BASE_URL = "https://huggingface.co/datasets/AthiyamanP/PatrolIQ-Chicago-Crime-Analytics/resolve/main/"
PROJECT_ROOT = Path(__file__).resolve().parent


def _source(filename: str):
    local = PROJECT_ROOT / "data" / "processed" / filename
    return local if local.exists() else HF_BASE_URL + filename


@st.cache_resource(show_spinner="Loading PatrolIQ data...")
def load_clustered_data():
    data = pd.read_parquet(
        _source("chicago_crime_clustered.parquet"),
        engine="pyarrow",
    )
    if "Date" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data


@st.cache_resource(show_spinner="Loading dimensionality-reduction data...")
def load_reduced_data():
    return pd.read_parquet(
        _source("chicago_crime_reduced.parquet"),
        engine="pyarrow",
    )
