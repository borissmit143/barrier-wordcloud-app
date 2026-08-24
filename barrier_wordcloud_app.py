"""Streamlit app: simulate one adoption barrier per demographic profile."""

from __future__ import annotations

import os
import re
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from wordcloud import STOPWORDS, WordCloud


APP_DIR = Path(__file__).resolve().parent
DEFAULT_USERS_FILE = APP_DIR / "users.xlsx"


class BarrierResponse(BaseModel):
    barrier: str = Field(
        description="Exactly one concise, specific barrier to adopting the service."
    )


def clean_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


@st.cache_data(show_spinner=False)
def load_default_demographics() -> pd.DataFrame:
    return pd.read_excel(DEFAULT_USERS_FILE)


def load_uploaded_demographics(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def make_prompt(
    firm_name: str,
    service: str,
    description: str,
    profile: dict[str, str],
) -> str:
    profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    return f"""You are role-playing a potential customer with this demographic profile:
{profile_lines}

Firm: {firm_name}
Service provided: {service}
Service description: {description}

From this customer's perspective, identify the single most important realistic
barrier that could prevent or delay adoption of the firm's service. Base the
answer on the entire demographic profile, but do not stereotype or invent facts.
Return exactly one concise barrier (one sentence, no list, no preamble)."""


def generate_barriers(
    demographics: pd.DataFrame,
    firm_name: str,
    service: str,
    description: str,
    api_key: str,
    model_name: str,
) -> pd.DataFrame:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.4,
        google_api_key=api_key,
    ).with_structured_output(BarrierResponse)

    records: list[dict[str, object]] = []
    progress = st.progress(0, text="Preparing customer profiles...")
    total = len(demographics)

    for position, (source_index, row) in enumerate(demographics.iterrows(), start=1):
        profile = {
            str(column): value
            for column, raw_value in row.items()
            if (value := clean_value(raw_value)) is not None
        }
        record: dict[str, object] = {"source_row": source_index + 2, **profile}
        prompt = make_prompt(firm_name, service, description, profile)

        for attempt in range(3):
            try:
                response = llm.invoke(prompt)
                record["Adoption Barrier"] = response.barrier.strip()
                record["Status"] = "Success"
                break
            except Exception as exc:  # Preserve the other customers if one call fails.
                if attempt < 2:
                    time.sleep(2**attempt)
                else:
                    record["Adoption Barrier"] = ""
                    record["Status"] = f"Error: {exc}"

        records.append(record)
        progress.progress(position / total, text=f"Customer {position} of {total}")

    progress.empty()
    return pd.DataFrame(records)


def wordcloud_figure(barriers: list[str], firm_name: str):
    text = " ".join(barriers)
    custom_stopwords = STOPWORDS.union(
        {
            "service", "firm", "company", "adopt", "adopting", "adoption",
            "customer", "customers", "could", "may", "might", "would",
        }
    )
    cloud = WordCloud(
        width=1500,
        height=750,
        background_color="white",
        colormap="viridis",
        stopwords=custom_stopwords,
        collocations=True,
        max_words=150,
        prefer_horizontal=0.85,
        random_state=42,
    ).generate(text)

    figure, axis = plt.subplots(figsize=(15, 7.5))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title(f"Barriers to adopting {firm_name}'s service", fontsize=18, pad=18)
    figure.tight_layout()
    return figure


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "firm"


def matching_column(dataframe: pd.DataFrame, wanted: str) -> str | None:
    """Find a demographic column without requiring exact capitalization."""
    normalized = {str(column).strip().casefold(): str(column) for column in dataframe.columns}
    return normalized.get(wanted.casefold())


def age_group_for(value: object) -> str | None:
    """Convert a numeric age into a standard, sortable customer age group."""
    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age) or age < 0:
        return None
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18–24"
    if age <= 34:
        return "25–34"
    if age <= 44:
        return "35–44"
    if age <= 54:
        return "45–54"
    if age <= 64:
        return "55–64"
    return "65+"


def filter_demographics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Render optional demographic filters and return matching customer rows."""
    age_column = matching_column(dataframe, "Age")
    gender_column = matching_column(dataframe, "Gender")
    location_column = matching_column(dataframe, "Location")

    st.subheader("Choose customers")
    filter1, filter2, filter3 = st.columns(3)
    mask = pd.Series(True, index=dataframe.index)

    if age_column:
        age_groups = dataframe[age_column].map(age_group_for)
        available_groups = [
            group
            for group in ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
            if group in set(age_groups.dropna())
        ]
        selected_age_groups = filter1.multiselect(
            "Age group",
            available_groups,
            placeholder="All age groups",
            help="Leave empty to include every age group.",
        )
        if selected_age_groups:
            mask &= age_groups.isin(selected_age_groups)
    else:
        filter1.caption("Age filter unavailable: no Age column found.")

    if location_column:
        locations = sorted(
            {value for raw in dataframe[location_column] if (value := clean_value(raw))},
            key=str.casefold,
        )
        selected_locations = filter2.multiselect(
            "Location",
            locations,
            placeholder="All locations",
            help="Leave empty to include every location.",
        )
        if selected_locations:
            mask &= dataframe[location_column].map(clean_value).isin(selected_locations)
    else:
        filter2.caption("Location filter unavailable: no Location column found.")

    if gender_column:
        genders = sorted(
            {value for raw in dataframe[gender_column] if (value := clean_value(raw))},
            key=str.casefold,
        )
        selected_genders = filter3.multiselect(
            "Gender",
            genders,
            placeholder="All genders",
            help="Leave empty to include every gender.",
        )
        if selected_genders:
            mask &= dataframe[gender_column].map(clean_value).isin(selected_genders)
    else:
        filter3.caption("Gender filter unavailable: no Gender column found.")

    return dataframe.loc[mask].copy()


def configured_api_key() -> str:
    """Read a key without failing when no Streamlit secrets file exists."""
    if environment_key := os.getenv("GOOGLE_API_KEY"):
        return environment_key
    try:
        return str(st.secrets.get("GOOGLE_API_KEY", ""))
    except Exception:
        return ""


st.set_page_config(page_title="Customer Adoption Barrier Word Cloud", layout="wide")
st.title("Customer Adoption Barrier Word Cloud")
st.caption(
    "Each row of demographic data becomes one simulated customer who provides "
    "one barrier to adopting the firm's service."
)

with st.sidebar:
    st.header("Model settings")
    api_key = st.text_input(
        "Google API key",
        value=configured_api_key(),
        type="password",
        help="Use GOOGLE_API_KEY in the environment or .streamlit/secrets.toml to avoid entering it here.",
    )
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash-lite")
    uploaded_file = st.file_uploader(
        "Optional demographics file",
        type=["xlsx", "xls", "csv"],
        help="Every column will be included in the customer profile.",
    )

st.subheader("Firm and service")
col1, col2 = st.columns(2)
firm_name = col1.text_input("Firm name", placeholder="e.g., Uber")
service = col2.text_input("Service provided", placeholder="e.g., On-demand ride hailing")
description = st.text_area(
    "Service description",
    placeholder="Describe what the service does, how it is delivered, and its intended customers.",
    height=120,
)

try:
    demographics = (
        load_uploaded_demographics(uploaded_file)
        if uploaded_file is not None
        else load_default_demographics()
    )
    demographics = demographics.dropna(how="all").reset_index(drop=True)
    filtered_demographics = filter_demographics(demographics)
    st.subheader("Demographic profiles")
    st.write(
        f"{len(filtered_demographics):,} of {len(demographics):,} customers selected · "
        f"{len(demographics.columns)} demographic fields"
    )
    st.dataframe(filtered_demographics, use_container_width=True, height=260)
except Exception as exc:
    st.error(f"Could not load the demographics file: {exc}")
    st.stop()

if st.button("Generate adoption barriers", type="primary", use_container_width=True):
    missing = [
        label
        for label, value in {
            "firm name": firm_name,
            "service provided": service,
            "service description": description,
            "Google API key": api_key,
        }.items()
        if not value.strip()
    ]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    elif filtered_demographics.empty:
        st.error("No customers match the selected filters.")
    else:
        with st.spinner("Simulating customer responses..."):
            st.session_state["barrier_results"] = generate_barriers(
                filtered_demographics, firm_name, service, description, api_key, model_name
            )
            st.session_state["result_firm"] = firm_name

if "barrier_results" in st.session_state:
    results = st.session_state["barrier_results"]
    result_firm = st.session_state["result_firm"]
    successful = results.loc[results["Status"] == "Success", "Adoption Barrier"].dropna()
    successful = [text for text in successful.astype(str) if text.strip()]

    st.subheader("Results")
    if successful:
        figure = wordcloud_figure(successful, result_firm)
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=200, bbox_inches="tight")
        stem = safe_filename(result_firm)
        csv_data = results.to_csv(index=False).encode("utf-8-sig")

        cloud_tab, responses_tab = st.tabs(["Word cloud", "Customer responses"])
        with cloud_tab:
            st.success(f"Word cloud created from {len(successful):,} customer responses.")
            st.pyplot(figure, use_container_width=True)
        with responses_tab:
            st.dataframe(results, use_container_width=True, height=480)

        plt.close(figure)
        download1, download2 = st.columns(2)
        download1.download_button(
            "Download responses (CSV)", csv_data, f"{stem}_adoption_barriers.csv", "text/csv"
        )
        download2.download_button(
            "Download word cloud (PNG)", image_buffer.getvalue(),
            f"{stem}_barrier_wordcloud.png", "image/png"
        )
    else:
        st.dataframe(results, use_container_width=True, height=480)
        st.error("No responses were generated. Check the Status column and your API/model settings.")
