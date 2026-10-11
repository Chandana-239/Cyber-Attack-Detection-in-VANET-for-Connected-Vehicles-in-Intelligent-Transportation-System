#!/usr/bin/env python3
"""Streamlit dashboard for VANET cyber-attack detection (simulation-trained models)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import (
    CANDIDATE_FEATURES,
    CLASS_LABELS,
    CLASS_NAMES,
    DATASET_PATH,
    EVALUATION_JSON_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    RF_PIPELINE_PATH,
    SELECTED_META_PATH,
    SVM_PIPELINE_PATH,
    TARGET_COLUMN,
    VALIDATION_JSON_PATH,
    class_name,
    load_raw_dataset,
    usable_feature_list,
)

st.set_page_config(
    page_title="VANET Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #123154 0%, #070b14 42%, #05070d 100%); color: #e8eefc; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0c1424 0%, #070b14 100%); border-right: 1px solid #1e2d4a; }
[data-testid="stHeader"] { background: rgba(7,11,20,0.6); }
h1, h2, h3 { color: #f4f8ff !important; letter-spacing: 0.02em; }
.hero-title { font-size: 1.55rem; font-weight: 700; color: #f4f8ff; line-height: 1.35; margin-bottom: 0.35rem; }
.hero-sub { color: #9db0d0; font-size: 0.95rem; margin-bottom: 1.1rem; }
.soc-card { background: linear-gradient(180deg, #121a2bcc 0%, #0d1524cc 100%); border: 1px solid #223356; border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
.kpi { background: #10182a; border: 1px solid #27406a; border-radius: 12px; padding: 0.9rem 1rem; }
.kpi .lbl { color: #8ea3c7; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi .val { color: #00e5ff; font-size: 1.55rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.warn { background: #2a1a12; border: 1px solid #a45a2a; color: #ffd7b0; border-radius: 10px; padding: 0.8rem 1rem; }
.ok { background: #10241c; border: 1px solid #1f6b4a; color: #b6f5d3; border-radius: 10px; padding: 0.8rem 1rem; }
.badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.75rem; background: #14324a; color: #00e5ff; border: 1px solid #1f5f86; }
.stMetric { background: #10182a; border: 1px solid #27406a; border-radius: 12px; padding: 0.6rem 0.8rem; }
.stButton>button { background: linear-gradient(90deg, #00b7c7, #0077aa); color: white; border: 0; font-weight: 600; }
div[data-testid="stDataFrame"] { background: #0d1524; }
</style>
"""
st.markdown(SOC_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_raw_dataset()


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def models_ready() -> dict:
    return {
        "random_forest": RF_PIPELINE_PATH.is_file(),
        "svm": SVM_PIPELINE_PATH.is_file(),
        "evaluation": EVALUATION_JSON_PATH.is_file(),
    }


def page_header() -> None:
    st.markdown(
        "<div class='hero-title'>Cyber Attack Detection in VANET for Connected Vehicles "
        "in Intelligent Transportation System</div>"
        "<div class='hero-sub'>Security operations view over NS-3.48 FlowMonitor simulation data "
        "(500 vehicle nodes). Historical experiment records — not a live network tap.</div>",
        unsafe_allow_html=True,
    )


def render_kpis(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f"<div class='kpi'><div class='lbl'>{label}</div><div class='val'>{value}</div></div>",
            unsafe_allow_html=True,
        )


def page_dashboard() -> None:
    page_header()
    try:
        df = cached_dataset()
        dataset_ok = True
        dataset_error = ""
    except Exception as exc:  # noqa: BLE001
        df = None
        dataset_ok = False
        dataset_error = str(exc)

    eval_data = load_json(EVALUATION_JSON_PATH)
    ready = models_ready()

    if dataset_ok and df is not None:
        n_classes = int(df[TARGET_COLUMN].nunique()) if TARGET_COLUMN in df.columns else 0
        render_kpis(
            [
                ("Simulation records", f"{len(df):,}"),
                ("Attack classes", str(n_classes)),
                ("Trained models", str(int(ready["random_forest"]) + int(ready["svm"]))),
                ("Measurement level", "Flow"),
            ]
        )
        st.caption("Counts above describe the stored NS-3 dataset, not live connected-vehicle traffic.")
    else:
        st.error(f"Could not load dataset: {dataset_error}")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Class distribution (historical simulation)")
        if dataset_ok and df is not None and TARGET_COLUMN in df.columns:
            dist = (
                df[TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .rename(index=lambda i: f"{int(i)} · {class_name(int(i))}")
            )
            st.bar_chart(dist, color="#00e5ff")
            st.dataframe(
                pd.DataFrame(
                    {
                        "class_id": dist.index.astype(str).str.split(" · ").str[0],
                        "class_name": dist.index.astype(str).str.split(" · ").str[-1],
                        "records": dist.values,
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning("Class distribution is unavailable until the dataset CSV can be read.")

    with right:
        st.subheader("Available trained models")
        st.write(f"Random Forest pipeline: {'ready' if ready['random_forest'] else 'not trained'}")
        st.write(f"SVM pipeline: {'ready' if ready['svm'] else 'not trained'}")
        st.write(f"Evaluation file: {'ready' if ready['evaluation'] else 'missing'}")
        if not ready["evaluation"]:
            st.markdown(
                "<div class='warn'>Models have not been trained yet. From the project root run "
                "<code>python train_models.py</code> then reload this page.</div>",
                unsafe_allow_html=True,
            )
        selected = load_json(SELECTED_META_PATH)
        if selected:
            st.markdown(
                f"<div class='ok'>Selected model: <b>{selected.get('model_key')}</b><br/>"
                f"{selected.get('reason')}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Held-out test metrics")
        if eval_data:
            rf = eval_data["models"]["random_forest"]
            svm = eval_data["models"]["svm"]
            metrics_df = pd.DataFrame(
                [
                    {
                        "Model": "Random Forest",
                        "Accuracy": rf["accuracy"],
                        "Macro F1": rf["macro_f1"],
                        "Weighted F1": rf["weighted_f1"],
                        "Macro precision": rf["macro_precision"],
                        "Macro recall": rf["macro_recall"],
                        "Train time (s)": rf["train_time_seconds"],
                    },
                    {
                        "Model": "SVM",
                        "Accuracy": svm["accuracy"],
                        "Macro F1": svm["macro_f1"],
                        "Weighted F1": svm["weighted_f1"],
                        "Macro precision": svm["macro_precision"],
                        "Macro recall": svm["macro_recall"],
                        "Train time (s)": svm["train_time_seconds"],
                    },
                ]
            )
            st.dataframe(metrics_df, hide_index=True, width="stretch")
            st.caption(
                "Metrics are from RandomSeed 22 held-out simulation runs. They are not training-set scores."
            )
        else:
            st.info("Train the models to populate actual evaluation metrics. No placeholder accuracy is shown.")

    st.subheader("Model comparison chart")
    chart_path = RESULTS_DIR / "model_comparison.png"
    if chart_path.is_file():
        st.image(str(chart_path), width="stretch")
    elif eval_data:
        chart_df = pd.DataFrame(
            {
                "Random Forest": [
                    eval_data["models"]["random_forest"]["accuracy"],
                    eval_data["models"]["random_forest"]["macro_f1"],
                    eval_data["models"]["random_forest"]["weighted_f1"],
                ],
                "SVM": [
                    eval_data["models"]["svm"]["accuracy"],
                    eval_data["models"]["svm"]["macro_f1"],
                    eval_data["models"]["svm"]["weighted_f1"],
                ],
            },
            index=["Accuracy", "Macro F1", "Weighted F1"],
        )
        st.bar_chart(chart_df)
    else:
        st.info("Comparison chart appears after training.")

    st.subheader("Dataset and evaluation warnings")
    validation = load_json(VALIDATION_JSON_PATH)
    warnings = []
    if validation:
        warnings.extend(validation.get("warnings", []))
    if eval_data:
        warnings.extend(eval_data.get("limitations", []))
    if not warnings:
        warnings.append("Run python train_models.py to generate the full validation report.")
    for item in warnings:
        st.markdown(f"- {item}")


def page_explorer() -> None:
    page_header()
    st.subheader("Dataset explorer")
    st.caption(f"Read-only view of `{DATASET_PATH}`. Filtering here does not modify the CSV.")
    try:
        df = cached_dataset()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load dataset: {exc}")
        return

    preview_n = st.slider("Preview rows", min_value=10, max_value=min(200, len(df)), value=25)
    class_filter = st.multiselect(
        "Filter by attack class (display only)",
        options=CLASS_LABELS,
        format_func=lambda i: f"{i} — {class_name(i)}",
        default=CLASS_LABELS,
    )
    view = df[df[TARGET_COLUMN].isin(class_filter)].copy() if TARGET_COLUMN in df.columns else df.copy()

    tabs = st.tabs(["Preview", "Columns", "Class distribution", "Missing values", "Descriptive statistics"])
    with tabs[0]:
        st.dataframe(view.head(preview_n), width="stretch")
        st.download_button(
            "Download original dataset CSV",
            data=DATASET_PATH.read_bytes(),
            file_name="vanet_500_dataset.csv",
            mime="text/csv",
        )
    with tabs[1]:
        info = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "non_null": [int(df[c].notna().sum()) for c in df.columns],
                "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }
        )
        st.dataframe(info, hide_index=True, width="stretch")
        st.write("Model features (measurements only):", usable_feature_list(df))
        st.write("Excluded identifiers / leakage-prone fields are not used for training.")
    with tabs[2]:
        if TARGET_COLUMN in df.columns:
            dist = df[TARGET_COLUMN].value_counts().sort_index()
            dist.index = [f"{i} · {class_name(int(i))}" for i in dist.index]
            st.bar_chart(dist, color="#ff8a4c")
    with tabs[3]:
        missing = df.isna().sum().rename("missing_count").to_frame()
        missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
        st.dataframe(missing, width="stretch")
        st.caption(
            "AverageDelayMs, AverageJitterMs, and ThroughputKbps are empty when a flow has no received packets. "
            "Those cells are genuine missing measurements, not zeros."
        )
    with tabs[4]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df[numeric_cols].describe().T, width="stretch")


def _parse_optional_float(raw: str, field: str) -> float | None:
    text = (raw or "").strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number or left blank.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{field} must be finite.")
    return value


def page_detect() -> None:
    page_header()
    st.subheader("Attack detection on a user-entered flow sample")
    st.caption(
        "Enter FlowMonitor-style measurements for one flow. The prediction comes from a model trained on "
        "NS-3.48 simulation data. It is not proof of a real-world attack and is not live VANET telemetry."
    )
    ready = models_ready()
    if not (ready["random_forest"] and ready["svm"]):
        st.markdown(
            "<div class='warn'>Trained pipelines were not found in <code>models/</code>. "
            "Run <code>python train_models.py</code> from the project root, then reload.</div>",
            unsafe_allow_html=True,
        )
        return

    eval_data = load_json(EVALUATION_JSON_PATH) or {}
    features = eval_data.get("features") or list(CANDIDATE_FEATURES)
    ranges = eval_data.get("feature_ranges", {})

    model_choice = st.radio("Model", options=["Random Forest", "SVM"], horizontal=True)
    st.markdown("**Network measurements**")
    c1, c2, c3 = st.columns(3)
    with c1:
        tx_packets = st.number_input("TxPackets", min_value=0.0, value=29.0, step=1.0)
        rx_packets = st.number_input("RxPackets", min_value=0.0, value=29.0, step=1.0)
        lost_packets = st.number_input("LostPackets", min_value=0.0, value=0.0, step=1.0)
    with c2:
        tx_bytes = st.number_input("TxBytes", min_value=0.0, value=15660.0, step=1.0)
        rx_bytes = st.number_input("RxBytes", min_value=0.0, value=15660.0, step=1.0)
        packet_loss = st.number_input("PacketLossRate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    with c3:
        delay_raw = st.text_input("AverageDelayMs (blank if unknown)", value="5.8")
        jitter_raw = st.text_input("AverageJitterMs (blank if unknown)", value="3.0")
        thr_raw = st.text_input("ThroughputKbps (blank if unknown)", value="22.35")

    if st.button("Detect Attack", type="primary"):
        warnings = []
        try:
            delay = _parse_optional_float(delay_raw, "AverageDelayMs")
            jitter = _parse_optional_float(jitter_raw, "AverageJitterMs")
            thr = _parse_optional_float(thr_raw, "ThroughputKbps")
        except ValueError as exc:
            st.error(str(exc))
            return

        if rx_packets == 0 and any(v is not None for v in (delay, jitter, thr)):
            warnings.append(
                "RxPackets is 0 but delay/jitter/throughput were provided. In this dataset those fields are missing when nothing is received."
            )
        if rx_packets > tx_packets:
            warnings.append("RxPackets is greater than TxPackets, which is unusual for a single flow.")
        if packet_loss > 100 or packet_loss < 0:
            st.error("PacketLossRate must be between 0 and 100.")
            return

        row = {
            "TxPackets": tx_packets,
            "RxPackets": rx_packets,
            "LostPackets": lost_packets,
            "TxBytes": tx_bytes,
            "RxBytes": rx_bytes,
            "PacketLossRate": packet_loss,
            "AverageDelayMs": delay,
            "AverageJitterMs": jitter,
            "ThroughputKbps": thr,
        }
        sample = pd.DataFrame([{k: row.get(k) for k in features}])

        for col in features:
            stats = ranges.get(col, {})
            value = sample.at[0, col]
            if pd.isna(value):
                continue
            lo, hi = stats.get("train_min"), stats.get("train_max")
            if lo is not None and hi is not None and (value < lo or value > hi):
                warnings.append(
                    f"{col}={value} is outside the training range [{lo:.4g}, {hi:.4g}]."
                )

        model_key = "random_forest" if model_choice == "Random Forest" else "svm"
        path = RF_PIPELINE_PATH if model_key == "random_forest" else SVM_PIPELINE_PATH
        try:
            pipeline = joblib.load(path)
            pred = int(pipeline.predict(sample)[0])
        except Exception as exc:  # noqa: BLE001
            st.error(f"The selected model could not produce a prediction: {exc}")
            return

        clf = pipeline.named_steps.get("clf")
        score_block = None
        if model_key == "random_forest" and hasattr(clf, "predict_proba"):
            proba = pipeline.predict_proba(sample)[0]
            score_block = {
                "label": "Random Forest vote share (uncalibrated; not a calibrated probability)",
                "values": {class_name(c): float(p) for c, p in zip(pipeline.classes_, proba)},
            }
        elif hasattr(clf, "decision_function"):
            scores = np.ravel(pipeline.decision_function(sample))
            if len(scores) == len(getattr(pipeline, "classes_", [])):
                score_block = {
                    "label": "SVM one-vs-rest decision scores (not probabilities)",
                    "values": {class_name(c): float(s) for c, s in zip(pipeline.classes_, scores)},
                }

        st.markdown(f"<span class='badge'>{model_choice}</span>", unsafe_allow_html=True)
        st.metric("Predicted class", f"{pred} — {class_name(pred)}")
        st.write("Model used:", model_choice)
        if score_block:
            st.write(score_block["label"])
            st.dataframe(
                pd.DataFrame(
                    [{"class": k, "score": v} for k, v in score_block["values"].items()]
                ),
                hide_index=True,
                width="stretch",
            )
        st.write("Input feature values")
        st.dataframe(sample, hide_index=True, width="stretch")
        for w in warnings:
            st.warning(w)
        st.info(
            "This output is a classification from a model fitted on simulated flow records. "
            "It does not by itself confirm that a real vehicle, RSU, or ITS service is under attack, "
            "and it does not explain causal packet-level behaviour."
        )


def page_performance() -> None:
    page_header()
    st.subheader("Model performance on held-out simulation data")
    eval_data = load_json(EVALUATION_JSON_PATH)
    if not eval_data:
        st.markdown(
            "<div class='warn'>No evaluation file found at "
            "<code>results/ml/evaluation.json</code>. Train models first with "
            "<code>python train_models.py</code>.</div>",
            unsafe_allow_html=True,
        )
        return

    split = eval_data.get("split", {})
    st.write(
        f"Split method: **{split.get('method')}**. "
        f"Train seeds {split.get('train_seeds')} ({split.get('n_train')} rows). "
        f"Test seeds {split.get('test_seeds')} ({split.get('n_test')} rows). "
        f"Legacy rows excluded: {split.get('excluded_legacy_rows')}."
    )
    st.caption(split.get("reason", ""))

    rf = eval_data["models"]["random_forest"]
    svm = eval_data["models"]["svm"]
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Random Forest (test)**")
        st.metric("Accuracy", f"{rf['accuracy']:.4f}")
        st.metric("Macro F1", f"{rf['macro_f1']:.4f}")
        st.metric("Weighted F1", f"{rf['weighted_f1']:.4f}")
        st.write(f"Macro precision {rf['macro_precision']:.4f} · Macro recall {rf['macro_recall']:.4f}")
        st.write(f"Training time {rf['train_time_seconds']:.3f} s")
    with m2:
        st.markdown("**SVM (test)**")
        st.metric("Accuracy", f"{svm['accuracy']:.4f}")
        st.metric("Macro F1", f"{svm['macro_f1']:.4f}")
        st.metric("Weighted F1", f"{svm['weighted_f1']:.4f}")
        st.write(f"Macro precision {svm['macro_precision']:.4f} · Macro recall {svm['macro_recall']:.4f}")
        st.write(f"Training time {svm['train_time_seconds']:.3f} s")

    c1, c2 = st.columns(2)
    rf_cm = RESULTS_DIR / "confusion_matrix_random_forest.png"
    svm_cm = RESULTS_DIR / "confusion_matrix_svm.png"
    with c1:
        st.markdown("**Random Forest confusion matrix**")
        if rf_cm.is_file():
            st.image(str(rf_cm), width="stretch")
    with c2:
        st.markdown("**SVM confusion matrix**")
        if svm_cm.is_file():
            st.image(str(svm_cm), width="stretch")

    st.markdown("**Per-class reports**")
    r1, r2 = st.columns(2)
    with r1:
        st.caption("Random Forest")
        rf_csv = RESULTS_DIR / "classification_report_random_forest.csv"
        if rf_csv.is_file():
            st.dataframe(pd.read_csv(rf_csv), hide_index=True, width="stretch")
    with r2:
        st.caption("SVM")
        svm_csv = RESULTS_DIR / "classification_report_svm.csv"
        if svm_csv.is_file():
            st.dataframe(pd.read_csv(svm_csv), hide_index=True, width="stretch")

    st.markdown("**Model comparison**")
    cmp_png = RESULTS_DIR / "model_comparison.png"
    if cmp_png.is_file():
        st.image(str(cmp_png), width="stretch")
    st.info(f"Recommended model: **{eval_data.get('recommended_model')}**. {eval_data.get('recommendation_reason')}")

    st.subheader("Evaluation methodology")
    st.markdown(
        """
- Target: `AttackType` (0 Normal, 1 DoS, 2 Sybil, 3 Blackhole).
- Features: TxPackets, RxPackets, LostPackets, TxBytes, RxBytes, PacketLossRate, AverageDelayMs, AverageJitterMs, ThroughputKbps.
- Identifiers, addresses, node IDs, ports, XML paths, and scenario names are excluded.
- Split: RandomSeed 1 and 11 for training; RandomSeed 22 held out for testing. Both models use the same split.
- Missing delay/jitter/throughput values are not stored as zeros. The pipeline imputes training medians and adds missingness indicators.
- SVM applies StandardScaler inside its pipeline. Random Forest does not scale features.
- Reported scores are test-set only.
        """
    )
    st.subheader("Limitations")
    for item in eval_data.get("limitations", []):
        st.markdown(f"- {item}")


def page_about() -> None:
    page_header()
    st.subheader("About the project")
    st.markdown(
        """
**Vehicular Ad Hoc Networks (VANETs)** let connected vehicles and roadside units exchange safety and traffic messages without relying on a single cellular hop. They are a building block of **Intelligent Transportation Systems (ITS)**, which coordinate vehicles, infrastructure, and control centres.

**Why detection matters.** Flooding (DoS), identity spoofing (Sybil), and forwarding refusal (Blackhole) can delay or suppress those messages. A classifier that reads network measurements is one research tool for spotting abnormal flow behaviour. It is not a deployed intrusion-prevention system.

**Role of NS-3.48.** The records in this workspace were produced with **NS-3.48** using 500 vehicle nodes plus one RSU, IEEE 802.11p, and FlowMonitor. Labels come from the configured scenario of each run, not from a live SOC feed.

**How the dataset was generated.** Each scenario runs as a separate simulation. FlowMonitor exports IP-level flows. Extraction writes `dataset/vanet_500_dataset.csv`. One row is one flow, not one vehicle and not one packet. About 741 genuine flow records are present.

**Classifiers.**
- **Random Forest** is an ensemble of decision trees on the measurement features (no scaling). Vote shares are shown as uncalibrated scores.
- **SVM** is an RBF-kernel classifier on a scaled, imputed feature matrix. Multiclass scores are one-vs-rest decision values, not probabilities.

**Classes studied**

| ID | Name | What the 500-vehicle simulation actually does |
|---:|---|---|
| 0 | Normal | Legitimate UDP to the RSU, including a hop via vehicle 497. |
| 1 | DoS | Same as Normal, plus a high-rate UDP sender (vehicle 499). |
| 2 | Sybil | Application-level extra UDP flows claiming additional identities (ports 6001–6005). The NS-3 node ID does not change. |
| 3 | Blackhole | Application-level drop on the deliberate forwarding path at vehicle 497. Direct-to-RSU flows are not dropped. |
        """
    )
    st.subheader("Limitations that apply today")
    st.markdown(
        """
- The **Sybil** implementation is application-level. It is not a protocol-level IEEE 1609 / pseudonym Sybil attack.
- The **Blackhole** implementation has known limitations and may not represent a routing-protocol Blackhole attack (no AODV blackhole).
- The dataset contains **simulated flow-level observations**. 500 vehicles do not yield 500 independent labelled rows.
- The dataset **alone does not establish real-world detection performance**.
- `AttackType` labels the **scenario**, so many rows in an attack run are ordinary UDP flows that happened during that run.
- Delay, jitter, and throughput are missing when a flow receives no packets; they are not silent zeros in the CSV.
- This dashboard does not execute shell commands, does not reconfigure NS-3, and does not overwrite the original dataset.
        """
    )
    st.caption(f"Project root: {PROJECT_ROOT} · Models directory: {MODELS_DIR}")


def main() -> None:
    st.sidebar.markdown("### VANET SOC")
    st.sidebar.caption("500-vehicle NS-3.48 experiment")
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Executive dashboard",
            "Dataset explorer",
            "Attack detection",
            "Model performance",
            "About the project",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Data source")
    st.sidebar.code("dataset/vanet_500_dataset.csv")
    st.sidebar.write("This application never treats CSV rows as live traffic.")

    if page == "Executive dashboard":
        page_dashboard()
    elif page == "Dataset explorer":
        page_explorer()
    elif page == "Attack detection":
        page_detect()
    elif page == "Model performance":
        page_performance()
    else:
        page_about()


main()
