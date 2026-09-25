"""
Network Anomaly Detector — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_generator import generate_dataset
from src.detectors import get_detector
from src.evaluator import evaluate

st.set_page_config(
    page_title="Network Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🛡️ Network Anomaly Detector")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["Demo (synthetic data)", "Upload CSV", "About"], index=0)
method = st.sidebar.selectbox(
    "Detection method",
    ["hybrid", "statistical", "isolation_forest", "rule_based"],
    index=0,
)

with st.sidebar.expander("Advanced settings"):
    z_threshold = st.slider("Statistical threshold (MAD Z)", 2.0, 6.0, 3.5, 0.1)
    contamination = st.slider("Isolation Forest contamination", 0.01, 0.20, 0.05, 0.01)
    min_votes = st.slider("Hybrid min votes", 1, 3, 2)


@st.cache_data
def load_or_generate(n_samples: int = 4000, ratio: float = 0.05, seed: int = 42):
    return generate_dataset(n_samples=n_samples, anomaly_ratio=ratio, seed=seed)


def run_detection(df: pd.DataFrame, method: str, z_threshold: float, contamination: float, min_votes: int):
    kwargs = {}
    if method == "statistical":
        kwargs["threshold"] = z_threshold
    elif method == "isolation_forest":
        kwargs["contamination"] = contamination
    elif method == "hybrid":
        kwargs["z_threshold"] = z_threshold
        kwargs["contamination"] = contamination
        kwargs["min_votes"] = min_votes
    detector = get_detector(method, **kwargs)
    return detector.fit_predict(df)


st.title("🛡️ Network Anomaly Detector")
st.caption("Statistical · Isolation Forest · Rule-based · Hybrid ensemble")

if mode == "About":
    st.markdown(
        """
        ### What this tool does
        Detects anomalous network flows using four complementary approaches:

        | Method | Strength |
        |--------|----------|
        | **Statistical (MAD)** | Fast, explainable |
        | **Isolation Forest** | Complex multi-feature outliers |
        | **Rule-based** | Explicit attack signatures |
        | **Hybrid** | Majority vote + score fusion |

        Built as part of a cybersecurity project scaffold.
        """
    )
    st.stop()

if mode == "Demo (synthetic data)":
    col_a, col_b = st.columns([2, 1])
    with col_a:
        n_samples = st.slider("Number of flows", 1000, 15000, 4000, 500)
    with col_b:
        anomaly_ratio = st.slider("True anomaly ratio", 0.01, 0.15, 0.05, 0.01)

    if st.button("Generate & Detect", type="primary"):
        with st.spinner("Generating synthetic traffic..."):
            df = load_or_generate(n_samples, anomaly_ratio)
            st.session_state["df"] = df
            st.session_state["has_labels"] = True
    elif "df" not in st.session_state:
        with st.spinner("Loading default dataset..."):
            df = load_or_generate(4000, 0.05)
            st.session_state["df"] = df
            st.session_state["has_labels"] = True
    else:
        df = st.session_state["df"]
else:
    uploaded = st.file_uploader("Upload network flow CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with columns such as: packet_count, byte_count, duration_sec, dst_port, syn_count, ...")
        st.stop()
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df
    st.session_state["has_labels"] = "is_anomaly" in df.columns

df = st.session_state["df"]
has_labels = st.session_state.get("has_labels", False)

with st.spinner(f"Running {method} detector..."):
    result = run_detection(df, method, z_threshold, contamination, min_votes)

df_view = df.copy()
df_view["anomaly"] = result.labels
df_view["anomaly_score"] = result.scores

n_anom = int(result.labels.sum())
pct = 100 * n_anom / len(df) if len(df) else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total flows", f"{len(df):,}")
k2.metric("Anomalies detected", f"{n_anom:,}", f"{pct:.1f}%")
k3.metric("Method", method.replace("_", " ").title())
if has_labels:
    metrics = evaluate(df["is_anomaly"].values, result.labels, result.scores)
    k4.metric("F1 (vs ground truth)", f"{metrics['f1']:.3f}")
else:
    k4.metric("Max score", f"{result.scores.max():.3f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Top Anomalies", "📈 Distributions", "📋 Raw Data"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        counts = pd.Series(result.labels).value_counts().reindex([0, 1], fill_value=0)
        fig = px.pie(
            names=["Normal", "Anomaly"], values=counts.values,
            color=["Normal", "Anomaly"],
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
            title="Detection breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            df_view, x="anomaly_score",
            color=df_view["anomaly"].map({0: "Normal", 1: "Anomaly"}),
            nbins=50, title="Anomaly score distribution",
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
            barmode="overlay", opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)
    if has_labels:
        st.subheader("Evaluation against ground truth")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        m2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        m3.metric("F1-score", f"{metrics.get('f1', 0):.3f}")
        auc = metrics.get("roc_auc")
        m4.metric("ROC-AUC", f"{auc:.3f}" if auc is not None else "—")

with tab2:
    st.subheader("Highest-scoring anomalous flows")
    top_n = st.slider("Show top N", 5, 50, 15)
    top = df_view[df_view["anomaly"] == 1].nlargest(top_n, "anomaly_score")
    show_cols = [c for c in ["timestamp", "src_ip", "dst_ip", "dst_port", "protocol",
        "packet_count", "byte_count", "packets_per_sec", "syn_count",
        "rst_count", "unique_dst_ports", "anomaly_score"] if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True)

with tab3:
    numeric = [c for c in ["packet_count", "byte_count", "packets_per_sec", "bytes_per_sec",
        "syn_count", "rst_count", "unique_dst_ports", "duration_sec"] if c in df_view.columns]
    if numeric:
        feat = st.selectbox("Feature", numeric)
        fig = px.box(
            df_view,
            x=df_view["anomaly"].map({0: "Normal", 1: "Anomaly"}),
            y=feat,
            color=df_view["anomaly"].map({0: "Normal", 1: "Anomaly"}),
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
            title=f"{feat} — Normal vs Anomaly",
            log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.dataframe(df_view.head(500), use_container_width=True)
    csv = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results CSV", csv, file_name="anomaly_results.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.caption("Network Anomaly Detector v1.0 · Cybersecurity Project")
