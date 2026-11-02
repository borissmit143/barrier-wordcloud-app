"""Streamlit app: simulate one adoption barrier per demographic profile."""

from __future__ import annotations

import os
import re
import time
import hashlib
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
MODEL_NAME = "gemini-3.1-flash-lite"


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
    customer_persona: str,
    profile: dict[str, str],
    perspective_lens: str,
    earlier_barriers: list[str],
) -> str:
    profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    earlier_text = (
        "\n".join(f"- {barrier}" for barrier in earlier_barriers[-12:])
        if earlier_barriers
        else "None yet."
    )
    return f"""You are role-playing a potential customer with this demographic profile:
{profile_lines}

Firm: {firm_name}
Service provided: {service}
Service description: {description}
Customer persona: {customer_persona}

Perspective lens to consider first: {perspective_lens}

Barriers already mentioned by other simulated customers:
{earlier_text}

From this customer's perspective, identify the single most important realistic
barrier that could prevent or delay adoption of the firm's service. Base the
answer on the entire demographic profile, but do not stereotype or invent facts.
Use the perspective lens only when it is genuinely plausible for this profile.
Avoid repeating an earlier barrier unless it is clearly the most credible answer;
if the underlying issue is similar, make the customer's specific reason distinct.
Return exactly one concise and concrete barrier (one sentence, no list, no preamble)."""


def perspective_lens(profile: dict[str, str]) -> str:
    """Assign profiles a stable lens to encourage credible response variety."""
    lenses = [
        "price, fees, or perceived value",
        "trust, credibility, or reputation",
        "content or service relevance",
        "ease of use or accessibility",
        "time, effort, or convenience",
        "privacy, security, or data use",
        "social, cultural, or values fit",
        "awareness or understanding of the service",
        "availability or geographic fit",
        "preference for existing alternatives",
    ]
    profile_key = "|".join(f"{key}={value}" for key, value in profile.items())
    digest = hashlib.sha256(profile_key.encode("utf-8")).digest()
    return lenses[int.from_bytes(digest[:4], "big") % len(lenses)]


def generate_barriers(
    demographics: pd.DataFrame,
    firm_name: str,
    service: str,
    description: str,
    customer_persona: str,
    api_key: str,
) -> pd.DataFrame:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.7,
        google_api_key=api_key,
    ).with_structured_output(BarrierResponse)

    records: list[dict[str, object]] = []
    completed_barriers: list[str] = []
    progress = st.progress(0, text="Preparing customer profiles...")
    total = len(demographics)

    for position, (source_index, row) in enumerate(demographics.iterrows(), start=1):
        profile = {
            str(column): value
            for column, raw_value in row.items()
            if (value := clean_value(raw_value)) is not None
        }
        record: dict[str, object] = {"source_row": source_index + 2, **profile}
        prompt = make_prompt(
            firm_name,
            service,
            description,
            customer_persona,
            profile,
            perspective_lens(profile),
            completed_barriers,
        )

        for attempt in range(3):
            try:
                response = llm.invoke(prompt)
                record["Adoption Barrier"] = response.barrier.strip()
                record["Status"] = "Success"
                completed_barriers.append(record["Adoption Barrier"])
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
    """Convert numeric ages or common age-range labels into a standard group."""
    if pd.isna(value):
        return None

    text = str(value).strip().casefold()
    normalized_text = re.sub(r"\s+", "", text).replace("–", "-").replace("—", "-")
    known_labels = {
        "under18": "Under 18",
        "<18": "Under 18",
        "lessthan18": "Under 18",
        "18-24": "18–24",
        "25-34": "25–34",
        "35-44": "35–44",
        "45-54": "45–54",
        "55-64": "55–64",
        "65+": "65+",
        "65andover": "65+",
        "65orolder": "65+",
    }
    label_without_years = normalized_text.replace("yearsold", "").replace("years", "")
    if label_without_years in known_labels:
        return known_labels[label_without_years]

    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age):
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers:
            age = float(numbers[0])
    if pd.isna(age) or age < 0:
        # For an unfamiliar category, preserve the file's label instead of
        # silently producing an empty age-filter menu.
        return str(value).strip() or None
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
        standard_groups = [
            group
            for group in ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
            if group in set(age_groups.dropna())
        ]
        other_groups = sorted(
            set(age_groups.dropna()) - set(standard_groups), key=str.casefold
        )
        available_groups = standard_groups + other_groups
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

    matching = dataframe.loc[mask].copy()
    if matching.empty:
        st.warning("No customers match the selected demographic filters.")
        return matching

    requested_count = st.number_input(
        "Number of customers to simulate",
        min_value=1,
        max_value=len(matching),
        value=len(matching),
        step=1,
        help=(
            "Choose how many of the matching demographic profiles should respond. "
            "When fewer are selected, the app takes a reproducible random sample."
        ),
    )
    requested_count = int(requested_count)
    if requested_count < len(matching):
        matching = matching.sample(n=requested_count, random_state=42).sort_index()
    return matching


def configured_api_key() -> str:
    """Read the Gemini key from Streamlit secrets (or the host environment)."""
    try:
        secret_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        if secret_key:
            return str(secret_key)
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")


st.set_page_config(page_title="Customer Adoption Barrier Word Cloud", layout="wide")
st.title("Customer Adoption Barrier Word Cloud")
st.caption(
    "Each row of demographic data becomes one simulated customer who provides "
    "one barrier to adopting the firm's service."
)

with st.sidebar:
    st.header("Model settings")
    api_key = configured_api_key()
    if api_key:
        st.success("Google API key is configured securely.")
    else:
        st.error(
            "Google API key not found. Add GOOGLE_API_KEY to Streamlit secrets."
        )
    st.caption(f"Model: `{MODEL_NAME}`")
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
customer_persona = st.text_area(
    "Customer persona",
    placeholder=(
        "Describe the target customer's goals, needs, behaviors, motivations, "
        "and concerns."
    ),
    help=(
        "This shared persona guides every simulation; each spreadsheet row adds "
        "that customer's demographic details."
    ),
    height=140,
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
            "customer persona": customer_persona,
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
                filtered_demographics,
                firm_name,
                service,
                description,
                customer_persona,
                api_key,
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
