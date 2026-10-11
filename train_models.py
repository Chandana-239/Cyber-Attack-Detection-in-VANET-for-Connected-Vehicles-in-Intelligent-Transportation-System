#!/usr/bin/env python3
"""Train Random Forest and SVM on the 500-vehicle VANET FlowMonitor dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_core import (
    CLASS_LABELS,
    CLASS_NAMES,
    DATASET_PATH,
    EVALUATION_JSON_PATH,
    MISSINGNESS_FEATURES,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PIPELINE_PATH,
    SELECTED_META_PATH,
    SELECTED_PIPELINE_PATH,
    SPLIT_GROUP_COLUMN,
    SVM_PIPELINE_PATH,
    TARGET_COLUMN,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_JSON_PATH,
    class_name,
    load_raw_dataset,
    prepare_ml_frame,
)


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(features: list[str], scale: bool) -> Pipeline:
    missing_cols = [c for c in MISSINGNESS_FEATURES if c in features]
    transformer = ColumnTransformer(
        transformers=[
            ("imputed", SimpleImputer(strategy="median"), features),
            (
                "missing",
                MissingIndicator(features="missing-only", error_on_new=False),
                missing_cols,
            ),
        ],
        remainder="drop",
    )
    steps = [("columns", transformer)]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_random_forest(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(features, scale=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_svm(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(features, scale=True)),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def seed_wise_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[SPLIT_GROUP_COLUMN].isin(TRAIN_SEEDS)].copy()
    test = df[df[SPLIT_GROUP_COLUMN].isin(TEST_SEEDS)].copy()
    if train.empty or test.empty:
        raise RuntimeError(
            "Seed-wise split produced an empty train or test set. "
            f"Train seeds={TRAIN_SEEDS}, test seeds={TEST_SEEDS}."
        )
    return train, test


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=[class_name(i) for i in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class_report,
        "confusion_matrix": cm.astype(int).tolist(),
        "n_test": int(len(y_true)),
    }


def save_confusion_matrix(cm: list[list[int]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="mako",
        xticklabels=[class_name(i) for i in CLASS_LABELS],
        yticklabels=[class_name(i) for i in CLASS_LABELS],
        ax=ax,
        linewidths=0.4,
        linecolor="#1c2740",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="#0b1220")
    plt.close(fig)


def save_comparison_chart(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    melted = frame.melt(id_vars="model", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.barplot(data=melted, x="metric", y="value", hue="model", ax=ax, palette=["#00e5ff", "#ff8a4c"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score on held-out seed 22")
    ax.set_xlabel("")
    ax.set_title("Random Forest vs SVM (test set)")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="#0b1220")
    plt.close(fig)


def per_class_csv(per_class: dict, path: Path) -> None:
    rows = []
    for label in CLASS_LABELS:
        name = class_name(label)
        stats = per_class.get(name, {})
        rows.append(
            {
                "class_id": label,
                "class_name": name,
                "precision": stats.get("precision"),
                "recall": stats.get("recall"),
                "f1": stats.get("f1-score"),
                "support": stats.get("support"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def recommend_model(rf_metrics: dict, svm_metrics: dict) -> tuple[str, str]:
    rf_key = (rf_metrics["macro_f1"], rf_metrics["weighted_f1"], rf_metrics["accuracy"])
    svm_key = (svm_metrics["macro_f1"], svm_metrics["weighted_f1"], svm_metrics["accuracy"])
    if rf_key > svm_key:
        return (
            "random_forest",
            "Random Forest had higher macro F1 (then weighted F1, then accuracy) on the held-out seed-22 test set.",
        )
    if svm_key > rf_key:
        return (
            "svm",
            "SVM had higher macro F1 (then weighted F1, then accuracy) on the held-out seed-22 test set.",
        )
    return (
        "random_forest",
        "Test metrics were tied on macro F1, weighted F1, and accuracy; Random Forest is kept as the selected model because it does not require feature scaling and exposes vote-share scores.",
    )


def smoke_predict(pipeline: Pipeline, features: list[str], test_x: pd.DataFrame) -> dict:
    sample = test_x.iloc[[0]][features]
    pred = int(pipeline.predict(sample)[0])
    out = {"predicted_class": pred, "class_name": class_name(pred)}
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba(sample)[0]
        out["score_type"] = "random_forest_vote_share_uncalibrated"
        out["scores"] = {class_name(c): float(p) for c, p in zip(pipeline.classes_, proba)}
    elif hasattr(clf, "decision_function"):
        scores = pipeline.decision_function(sample)[0]
        out["score_type"] = "svm_ovr_decision_score"
        out["scores"] = {class_name(c): float(s) for c, s in zip(pipeline.classes_, scores)}
    return out


def main() -> int:
    _ensure_dirs()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.is_file():
        print(f"ERROR: dataset file does not exist: {DATASET_PATH}")
        return 1

    df = load_raw_dataset()
    model_df, features, meta = prepare_ml_frame(df)
    VALIDATION_JSON_PATH.write_text(json.dumps(meta["validation"], indent=2), encoding="utf-8")

    print(f"Records in CSV: {len(df)}")
    print(f"Features: {features}")
    print(f"Validation warnings: {len(meta['validation']['warnings'])}")
    for w in meta["validation"]["warnings"]:
        print(f"  - {w}")

    train_df, test_df = seed_wise_split(model_df)
    x_train = train_df[features]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    x_test = test_df[features]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    split_counts = {
        "n_csv_rows": int(len(df)),
        "n_used_for_ml": int(len(model_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_class_counts": {str(k): int(v) for k, v in train_df[TARGET_COLUMN].value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in test_df[TARGET_COLUMN].value_counts().sort_index().items()},
        "train_run_ids": sorted(int(x) for x in train_df["RunID"].unique()),
        "test_run_ids": sorted(int(x) for x in test_df["RunID"].unique()),
    }
    print(f"Split: train={split_counts['n_train']} test={split_counts['n_test']}")
    print(f"Train class counts: {split_counts['train_class_counts']}")
    print(f"Test class counts: {split_counts['test_class_counts']}")

    limitations = [
        "Evaluation uses a single held-out RandomSeed (22), not k-fold cross-validation.",
        "Test size is 246 flow records from four NS-3 runs. This is small for claiming operational performance.",
        "Sybil is application-level extra UDP identities; Blackhole is an application relay drop, not a routing-protocol blackhole.",
        "Scenario labels apply to every flow in a run, including non-attacker flows.",
        "Missing delay/jitter/throughput values are imputed from training medians with missingness indicators; they are not replaced with zeros in the CSV.",
    ]
    if split_counts["n_test"] < 300:
        limitations.append(
            "The held-out test set is smaller than 300 records. Per-class scores can move substantially with additional seeds."
        )

    feature_ranges = {
        col: {
            "train_min": None if pd.isna(x_train[col].min()) else float(x_train[col].min()),
            "train_max": None if pd.isna(x_train[col].max()) else float(x_train[col].max()),
            "train_median": None if pd.isna(x_train[col].median()) else float(x_train[col].median()),
            "n_missing_train": int(x_train[col].isna().sum()),
            "n_missing_test": int(x_test[col].isna().sum()),
        }
        for col in features
    }

    models = {
        "random_forest": build_random_forest(features),
        "svm": build_svm(features),
    }
    metrics_by_model: dict[str, dict] = {}
    train_times: dict[str, float] = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_times[name] = time.perf_counter() - t0
        y_pred = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["train_time_seconds"] = float(train_times[name])
        metrics_by_model[name] = metrics
        print(
            f"  test accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f} "
            f"train_s={metrics['train_time_seconds']:.3f}"
        )

    joblib.dump(models["random_forest"], RF_PIPELINE_PATH)
    joblib.dump(models["svm"], SVM_PIPELINE_PATH)
    print(f"Saved {RF_PIPELINE_PATH}")
    print(f"Saved {SVM_PIPELINE_PATH}")

    sns.set_theme(style="darkgrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1220",
            "axes.facecolor": "#121a2b",
            "axes.edgecolor": "#8ea0c0",
            "axes.labelcolor": "#e8eefc",
            "xtick.color": "#e8eefc",
            "ytick.color": "#e8eefc",
            "text.color": "#e8eefc",
            "savefig.facecolor": "#0b1220",
        }
    )
    save_confusion_matrix(
        metrics_by_model["random_forest"]["confusion_matrix"],
        "Random Forest confusion matrix (seed-22 test)",
        RESULTS_DIR / "confusion_matrix_random_forest.png",
    )
    save_confusion_matrix(
        metrics_by_model["svm"]["confusion_matrix"],
        "SVM confusion matrix (seed-22 test)",
        RESULTS_DIR / "confusion_matrix_svm.png",
    )
    comparison_rows = [
        {
            "model": "Random Forest",
            "Accuracy": metrics_by_model["random_forest"]["accuracy"],
            "Macro F1": metrics_by_model["random_forest"]["macro_f1"],
            "Weighted F1": metrics_by_model["random_forest"]["weighted_f1"],
            "Macro Precision": metrics_by_model["random_forest"]["macro_precision"],
            "Macro Recall": metrics_by_model["random_forest"]["macro_recall"],
        },
        {
            "model": "SVM",
            "Accuracy": metrics_by_model["svm"]["accuracy"],
            "Macro F1": metrics_by_model["svm"]["macro_f1"],
            "Weighted F1": metrics_by_model["svm"]["weighted_f1"],
            "Macro Precision": metrics_by_model["svm"]["macro_precision"],
            "Macro Recall": metrics_by_model["svm"]["macro_recall"],
        },
    ]
    save_comparison_chart(comparison_rows, RESULTS_DIR / "model_comparison.png")
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    per_class_csv(
        metrics_by_model["random_forest"]["per_class"],
        RESULTS_DIR / "classification_report_random_forest.csv",
    )
    per_class_csv(metrics_by_model["svm"]["per_class"], RESULTS_DIR / "classification_report_svm.csv")

    selected_key, selected_reason = recommend_model(metrics_by_model["random_forest"], metrics_by_model["svm"])
    selected_pipe = models[selected_key]
    joblib.dump(selected_pipe, SELECTED_PIPELINE_PATH)

    rf_loaded = joblib.load(RF_PIPELINE_PATH)
    svm_loaded = joblib.load(SVM_PIPELINE_PATH)
    smoke = {
        "random_forest": smoke_predict(rf_loaded, features, x_test),
        "svm": smoke_predict(svm_loaded, features, x_test),
    }
    print("\nSmoke prediction on first held-out test row:")
    print(json.dumps(smoke, indent=2))

    evaluation = {
        "dataset_path": str(DATASET_PATH),
        "features": features,
        "class_names": CLASS_NAMES,
        "split": {**meta["split"], **split_counts},
        "limitations": limitations,
        "feature_ranges": feature_ranges,
        "models": {
            "random_forest": {
                "pipeline_path": str(RF_PIPELINE_PATH),
                "score_type": "random_forest_vote_share_uncalibrated",
                **metrics_by_model["random_forest"],
            },
            "svm": {
                "pipeline_path": str(SVM_PIPELINE_PATH),
                "score_type": "svm_ovr_decision_score",
                **metrics_by_model["svm"],
            },
        },
        "recommended_model": selected_key,
        "recommendation_reason": selected_reason,
        "plots": {
            "confusion_matrix_random_forest": str(RESULTS_DIR / "confusion_matrix_random_forest.png"),
            "confusion_matrix_svm": str(RESULTS_DIR / "confusion_matrix_svm.png"),
            "model_comparison": str(RESULTS_DIR / "model_comparison.png"),
        },
        "evaluated_on": "held_out_random_seed_22_test_set",
        "not_evaluated_on": "training_set",
        "smoke_prediction": smoke,
    }
    EVALUATION_JSON_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    SELECTED_META_PATH.write_text(
        json.dumps(
            {
                "model_key": selected_key,
                "pipeline_path": str(SELECTED_PIPELINE_PATH),
                "reason": selected_reason,
                "test_macro_f1": metrics_by_model[selected_key]["macro_f1"],
                "test_accuracy": metrics_by_model[selected_key]["accuracy"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_rows = []
    for key, label in (("random_forest", "Random Forest"), ("svm", "SVM")):
        m = metrics_by_model[key]
        summary_rows.append(
            {
                "model": label,
                "accuracy": m["accuracy"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "train_time_seconds": m["train_time_seconds"],
                "n_test": m["n_test"],
                "selected": key == selected_key,
            }
        )
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print(f"\nRecommended model: {selected_key}")
    print(selected_reason)
    print(f"Evaluation written to {EVALUATION_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
