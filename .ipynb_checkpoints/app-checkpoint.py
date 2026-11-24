"""Streamlit front end (step 8): type a URL, get Legitimate / Phishing and a probability.

Run:  RESULTS_DIR=results streamlit run app.py
Requires the files written by `phishing_pipeline.py exp4`: results/models/char_cnn.keras and meta.json.
"""
import json
import os
import time
from pathlib import Path

import streamlit as st

from phishing_pipeline import predict_urls, normalise_url

RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    meta = json.load(open(RESULTS / "models" / "meta.json"))
    model = tf.keras.models.load_model(RESULTS / "models" / meta["model"])
    predict_urls(model, ["example.com"])            # warm-up so the first user request is not slow
    return model, meta


st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")
st.title("AI-Powered Phishing URL Detection")
st.caption("Character-level CNN on the URL string only. The result is decision support, not a guarantee.")

try:
    model, meta = load_model()
except Exception as e:                                # model not trained yet
    st.error(f"Model not found in {RESULTS}/models. Run `python phishing_pipeline.py exp4` first. ({e})")
    st.stop()

mode = st.radio("Decision threshold", ["Balanced (0.50)", "Recall-oriented (recall >= 99% on validation data)"], horizontal=False)
thr = 0.5 if mode.startswith("Balanced") else float(meta.get("threshold_recall_99", 0.5))
url = st.text_input("Enter a URL", placeholder="http://secure-login.example-bank.com/verify?id=1")

if st.button("Check URL") and url.strip():
    t0 = time.time()
    p = float(predict_urls(model, [url.strip()])[0])
    ms = 1000 * (time.time() - t0)
    if p >= thr:
        st.error(f"Phishing  (probability {p:.1%})")
    else:
        st.success(f"Legitimate  (phishing probability {p:.1%})")
    st.write(f"Checked as: `{normalise_url(url)}`  |  threshold {thr:.2f}  |  {ms:.1f} ms")
    st.info("Only the URL text is analysed. A benign-looking URL on a compromised site, or a very short URL, can be misclassified.")
