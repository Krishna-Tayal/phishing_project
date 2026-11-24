from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page and paths
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="PhiUSIIL | Phishing URL Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_DIR / "results_cpu_final")))
DATA_PATH = Path(
    os.getenv(
        "PHIUSIIL_CSV",
        str(PROJECT_DIR / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"),
    )
)
if not DATA_PATH.exists():
    root_csv = PROJECT_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if root_csv.exists():
        DATA_PATH = root_csv

CLEAN_DATA_PATH = RESULTS_DIR / "clean_PhiUSIIL.csv"
MODEL_PATH = RESULTS_DIR / "models" / "char_cnn.keras"
META_PATH = RESULTS_DIR / "models" / "meta.json"

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(115deg, rgba(35,75,125,.20), rgba(30,150,125,.10));
        border: 1px solid rgba(128,128,128,.22); margin-bottom: 1.1rem;
      }
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {margin: 0; opacity: .78;}
      .small-muted {opacity: .7; font-size: .86rem;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.2);
        padding: .8rem 1rem; border-radius: 12px;
        background: rgba(128,128,128,.035);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Cached data/model loading
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def read_json_cached(path: str, modified: float) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner="Loading the trained character-level CNN...")
def load_cnn_model(model_path: str, model_modified: float):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def load_dataset() -> tuple[pd.DataFrame | None, Path | None, str | None]:
    path = first_existing([CLEAN_DATA_PATH, DATA_PATH])
    if path is None:
        return None, None, (
            "Dataset not found. Expected either "
            f"{CLEAN_DATA_PATH} or {DATA_PATH}."
        )
    try:
        frame = read_csv_cached(str(path), path.stat().st_mtime)
        return frame, path, None
    except Exception as exc:
        return None, path, f"Could not read dataset: {exc}"


def load_optional_csv(filename: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        return read_csv_cached(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None


def get_class_counts(frame: pd.DataFrame) -> pd.Series | None:
    if "y" in frame.columns:
        labels = pd.to_numeric(frame["y"], errors="coerce")
        # Pipeline convention: y=0 benign, y=1 phishing.
        return pd.Series({
            "Legitimate": int((labels == 0).sum()),
            "Phishing": int((labels == 1).sum()),
        })
    if "label" in frame.columns:
        labels = pd.to_numeric(frame["label"], errors="coerce")
        # Raw PhiUSIIL convention: label=1 legitimate, label=0 phishing.
        return pd.Series({
            "Legitimate": int((labels == 1).sum()),
            "Phishing": int((labels == 0).sum()),
        })
    return None


def show_plot(filename: str, caption: str) -> None:
    path = RESULTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} has not been generated yet.")


def find_url_column(columns) -> str | None:
    lookup = {str(col).strip().lower(): col for col in columns}
    for candidate in ("url", "url_raw", "website", "link", "uri"):
        if candidate in lookup:
            return lookup[candidate]
    return None


def get_model_and_meta():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {META_PATH}")
    metadata = read_json_cached(str(META_PATH), META_PATH.stat().st_mtime)
    model = load_cnn_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    return model, metadata


def predict_batch(model, urls: list[str]):
    # Reuse the exact URL normalization and encoding implementation from the
    # project's existing training pipeline; do not recreate preprocessing here.
    from phishing_pipeline_cpu import predict_urls
    return predict_urls(model, urls)


# ---------------------------------------------------------------------
# Load available project outputs
# ---------------------------------------------------------------------
dataset, dataset_path, dataset_error = load_dataset()
exp1 = load_optional_csv("exp1.csv")
exp3 = load_optional_csv("exp3.csv")
exp4 = load_optional_csv("exp4.csv")
stats_path = RESULTS_DIR / "data_stats.json"
stats = (
    read_json_cached(str(stats_path), stats_path.stat().st_mtime)
    if stats_path.exists()
    else None
)

# ---------------------------------------------------------------------
# Sidebar navigation and status
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ PhiUSIIL")
    st.caption("Phishing URL intelligence dashboard")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Dataset Explorer", "Experiments", "URL Scanner"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Project resources**")
    st.caption(f"Results folder: `{RESULTS_DIR.name}`")
    st.caption("Dataset: PhiUSIIL")
    st.caption("Inference: character-level CNN")
    st.divider()
    if MODEL_PATH.exists() and META_PATH.exists():
        st.success("Model files found")
    else:
        st.warning("Model files not found")


st.markdown(
    """
    <div class="hero">
      <h1>Phishing URL Detection</h1>
      <p>Character-level deep learning on the PhiUSIIL dataset · CPU-compatible pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
if page == "Overview":
    st.subheader("Project overview")
    if dataset is not None:
        counts = get_class_counts(dataset)
        total = len(dataset)
        c1, c2, c3 = st.columns(3)
        c1.metric("URLs in loaded dataset", f"{total:,}")
        if counts is not None:
            c2.metric("Legitimate URLs", f"{counts['Legitimate']:,}")
            c3.metric("Phishing URLs", f"{counts['Phishing']:,}")
        else:
            c2.metric("Dataset columns", len(dataset.columns))
            c3.metric("Dataset source", "PhiUSIIL")
        with st.expander("View dataset sample"):
            st.dataframe(dataset.head(10), use_container_width=True)
    else:
        st.warning(dataset_error or "Dataset is not available.")

    st.subheader("Saved experiment status")
    status_cols = st.columns(3)
    for col, label, result in [
        (status_cols[0], "EXP1", exp1),
        (status_cols[1], "EXP3", exp3),
        (status_cols[2], "EXP4", exp4),
    ]:
        rows = len(result) if result is not None else 0
        col.metric(f"{label} result rows", rows)

    if exp4 is not None and not exp4.empty:
        st.markdown("**Proposed CNN evaluation — saved results**")
        st.dataframe(exp4, use_container_width=True)
    else:
        st.info("EXP4 results are not present yet. No performance values are being estimated.")

    left, right = st.columns(2)
    with left:
        show_plot("confusion_PhiUSIIL.png", "CNN confusion matrix")
    with right:
        show_plot("curves_PhiUSIIL.png", "Training and validation curves")

    if stats and isinstance(stats, dict):
        with st.expander("Preparation statistics"):
            st.json(stats.get("PhiUSIIL", stats))


# ---------------------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.subheader("PhiUSIIL dataset explorer")
    if dataset is None:
        st.error(dataset_error or "Dataset unavailable.")
    else:
        st.caption(f"Loaded from: {dataset_path}")
        counts = get_class_counts(dataset)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(dataset):,}")
        c2.metric("Columns", len(dataset.columns))

        if counts is not None:
            st.markdown("**Class distribution**")
            chart_data = counts.rename_axis("Class").reset_index(name="URLs")
            st.bar_chart(chart_data.set_index("Class"))

        search = st.text_input(
            "Search across displayed fields",
            placeholder="Search URL, domain, label...",
        )
        filtered = dataset
        if search.strip():
            mask = dataset.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False, regex=False)
            ).any(axis=1)
            filtered = dataset.loc[mask]

        st.caption(f"Showing {len(filtered):,} of {len(dataset):,} records")
        st.dataframe(filtered, use_container_width=True, height=520)
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="phiusiil_filtered.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------
# EXPERIMENTS AND MODEL
# ---------------------------------------------------------------------
elif page == "Experiments":
    st.subheader("Experiment results")
    for title, frame in [
        ("EXP1 · Classical models across FG1–FG4", exp1),
        ("EXP3 · Deep learning models on FG2", exp3),
        ("EXP4 · Proposed CNN evaluation", exp4),
    ]:
        with st.expander(title, expanded=(title.startswith("EXP4"))):
            if frame is not None and not frame.empty:
                st.dataframe(frame, use_container_width=True)
            else:
                st.info("No saved result file found for this experiment.")

    st.divider()
    st.subheader("Feature-group analysis")
    col1, col2 = st.columns(2)
    with col1:
        show_plot("accuracy_by_feature_group.png", "Validation accuracy across feature groups")
    with col2:
        show_plot("top2_feature_groups_accuracy.png", "Top two feature groups selected using validation accuracy")

    for fg in ("FG1", "FG2", "FG3", "FG4"):
        with st.expander(f"{fg} · model validation accuracy"):
            show_plot(f"{fg}/accuracy_by_model.png", f"{fg} validation accuracy")

    st.divider()
    st.subheader("Trained character-level CNN")
    if MODEL_PATH.exists() and META_PATH.exists():
        st.success("Model artifact and metadata are present.")
        st.caption(f"Model file: {MODEL_PATH}")
        if st.button("Load model and inspect metadata", type="primary"):
            try:
                model, metadata = get_model_and_meta()
                st.session_state["cnn_ready"] = True
                st.success("Model loaded successfully.")
                st.json(metadata)
                st.caption(f"Input shape: {model.input_shape} · Output shape: {model.output_shape}")
            except Exception as exc:
                st.session_state["cnn_ready"] = False
                st.error(f"Model loading failed: {exc}")
                st.exception(exc)
        if st.session_state.get("cnn_ready"):
            st.info("The CNN is loaded and available to the URL Scanner.")
    else:
        st.warning("The trained CNN or its metadata is missing.")
        st.write("Expected model:", str(MODEL_PATH))
        st.write("Expected metadata:", str(META_PATH))


# ---------------------------------------------------------------------
# URL SCANNER
# ---------------------------------------------------------------------
elif page == "URL Scanner":
    st.subheader("URL scanner")
    st.write("Score a single URL or upload a CSV for batch inference.")
    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.error(
            "The trained CNN files are missing. Run the pipeline's EXP4 training "
            "command first; this dashboard does not train or fabricate a model."
        )
    else:
        try:
            model, metadata = get_model_and_meta()
            st.success("Trained CNN loaded and ready.")
            default_threshold = float(metadata.get("default_threshold", 0.5))
            recall_threshold = float(metadata.get("threshold_recall_99", default_threshold))

            threshold_mode = st.radio(
                "Classification threshold",
                ["Default threshold", "99% recall threshold", "Custom threshold"],
                horizontal=True,
            )
            if threshold_mode == "Default threshold":
                threshold = default_threshold
            elif threshold_mode == "99% recall threshold":
                threshold = recall_threshold
            else:
                threshold = st.slider(
                    "Custom phishing threshold",
                    min_value=0.01,
                    max_value=0.99,
                    value=default_threshold,
                    step=0.01,
                )
            st.caption(
                f"Current threshold: {threshold:.2f}. A score at or above this "
                "value is labelled phishing. Threshold choice changes the "
                "classification, not the model probability."
            )

            single_tab, bulk_tab = st.tabs(["Single URL", "Bulk CSV"])
            with single_tab:
                url_value = st.text_input(
                    "Website URL",
                    placeholder="https://example.com/account/login",
                    key="single_url_input",
                )
                if st.button("Analyze URL", type="primary", key="single_scan"):
                    if not url_value.strip():
                        st.warning("Enter a URL to analyze.")
                    else:
                        try:
                            with st.spinner("Analyzing URL with the trained CNN..."):
                                probability = float(predict_batch(model, [url_value])[0])
                            prediction = "Phishing" if probability >= threshold else "Legitimate"
                            st.session_state["last_single_result"] = {
                                "url": url_value,
                                "probability": probability,
                                "prediction": prediction,
                                "threshold": threshold,
                            }
                        except Exception as exc:
                            st.error(f"Inference failed: {exc}")
                            st.exception(exc)

                result = st.session_state.get("last_single_result")
                if result:
                    st.divider()
                    st.markdown("**Latest analysis**")
                    st.code(result["url"], language=None)
                    score_col, verdict_col = st.columns(2)
                    score_col.metric("Phishing probability", f"{result['probability']:.2%}")
                    verdict_col.metric("Classification", result["prediction"])
                    st.progress(min(max(result["probability"], 0.0), 1.0))
                    if result["prediction"] == "Phishing":
                        st.error("Potential phishing URL — avoid entering credentials or sensitive information.")
                    else:
                        st.success("Classified as legitimate by the model. This is not a guarantee of safety.")
                    st.caption(
                        f"Threshold used: {result['threshold']:.2f}. "
                        "Predictions are model-based assessments, not a live website reputation check."
                    )

            with bulk_tab:
                uploaded = st.file_uploader(
                    "Upload a CSV with a URL column",
                    type=["csv"],
                    key="bulk_csv",
                )
                if uploaded is not None:
                    try:
                        input_frame = pd.read_csv(uploaded)
                        url_col = find_url_column(input_frame.columns)
                        if url_col is None:
                            st.error("No URL column found. Accepted names: url, url_raw, website, link, uri.")
                        else:
                            st.caption(f"Detected URL column: {url_col}")
                            st.dataframe(input_frame.head(8), use_container_width=True)
                            if len(input_frame) > 10000:
                                st.warning("Large upload: scanning may take some time on CPU.")
                            if st.button("Scan uploaded URLs", type="primary", key="bulk_scan"):
                                url_values = input_frame[url_col].fillna("").astype(str).tolist()
                                valid_indices = [i for i, value in enumerate(url_values) if value.strip()]
                                if not valid_indices:
                                    st.warning("The uploaded file contains no non-empty URLs.")
                                else:
                                    with st.spinner(f"Scanning {len(valid_indices):,} URLs..."):
                                        probabilities = predict_batch(
                                            model, [url_values[i] for i in valid_indices]
                                        )
                                    output = input_frame.copy()
                                    output["phishing_probability"] = pd.NA
                                    output["prediction"] = "Empty URL"
                                    for index, probability in zip(valid_indices, probabilities):
                                        probability = float(probability)
                                        output.at[index, "phishing_probability"] = probability
                                        output.at[index, "prediction"] = (
                                            "Phishing" if probability >= threshold else "Legitimate"
                                        )
                                    st.session_state["bulk_results"] = output
                                    st.session_state["bulk_threshold"] = threshold
                    except Exception as exc:
                        st.error(f"Could not process uploaded CSV: {exc}")
                        st.exception(exc)

                bulk_result = st.session_state.get("bulk_results")
                if bulk_result is not None:
                    st.divider()
                    st.markdown("**Batch prediction results**")
                    scanned = bulk_result["prediction"].isin(["Phishing", "Legitimate"])
                    phishing_count = int((bulk_result["prediction"] == "Phishing").sum())
                    legitimate_count = int((bulk_result["prediction"] == "Legitimate").sum())
                    a, b, c = st.columns(3)
                    a.metric("URLs scanned", int(scanned.sum()))
                    b.metric("Phishing classifications", phishing_count)
                    c.metric("Legitimate classifications", legitimate_count)
                    st.dataframe(bulk_result, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        data=bulk_result.to_csv(index=False).encode("utf-8"),
                        file_name="phiusiil_url_scan_results.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not initialize the scanner: {exc}")
            st.exception(exc)

st.divider()
st.caption(
    "Research prototype · PhiUSIIL dataset · Character-level CNN · "
    "A model classification is not a guarantee that a website is safe."
)
