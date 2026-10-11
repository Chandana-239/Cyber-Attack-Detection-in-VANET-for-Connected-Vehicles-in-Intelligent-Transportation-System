"""Shared paths, feature policy, and helpers for VANET ML training and the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "vanet_500_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "ml"

RF_PIPELINE_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
SVM_PIPELINE_PATH = MODELS_DIR / "svm_pipeline.joblib"
SELECTED_PIPELINE_PATH = MODELS_DIR / "selected_pipeline.joblib"
SELECTED_META_PATH = MODELS_DIR / "selected_model_meta.json"
EVALUATION_JSON_PATH = RESULTS_DIR / "evaluation.json"
VALIDATION_JSON_PATH = RESULTS_DIR / "dataset_validation.json"

CANDIDATE_FEATURES = [
    "TxPackets",
    "RxPackets",
    "LostPackets",
    "TxBytes",
    "RxBytes",
    "PacketLossRate",
    "AverageDelayMs",
    "AverageJitterMs",
    "ThroughputKbps",
]

IDENTIFIER_OR_LEAKAGE_COLUMNS = [
    "FlowID",
    "RunID",
    "RandomSeed",
    "Scenario",
    "AttackType",
    "SourceAddress",
    "DestinationAddress",
    "SourceNode",
    "DestinationNode",
    "Protocol",
    "SourcePort",
    "DestinationPort",
    "SourceXml",
    "MeasurementLevel",
]

TARGET_COLUMN = "AttackType"
SPLIT_GROUP_COLUMN = "RandomSeed"

CLASS_NAMES = {
    0: "Normal",
    1: "DoS Attack",
    2: "Sybil Attack",
    3: "Blackhole Attack",
}

CLASS_LABELS = [0, 1, 2, 3]
RANDOM_STATE = 42
TRAIN_SEEDS = [1.0, 11.0]
TEST_SEEDS = [22.0]

MISSINGNESS_FEATURES = ["AverageDelayMs", "AverageJitterMs", "ThroughputKbps"]


def class_name(label: int) -> str:
    return CLASS_NAMES.get(int(label), f"Unknown ({label})")


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else DATASET_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def sanitize_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert infinities to NaN. Do not fill missing measurements with zeros."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out


def usable_feature_list(df: pd.DataFrame) -> list[str]:
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    return features


def build_validation_report(df: pd.DataFrame) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    if TARGET_COLUMN not in df.columns:
        errors.append(f"Required target column '{TARGET_COLUMN}' is missing.")

    missing_features = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing_features:
        errors.append(f"Expected measurement columns are missing: {missing_features}")

    features = usable_feature_list(df)
    class_counts: dict[str, int] = {}
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
        class_counts = {str(int(k) if pd.notna(k) else k): int(v) for k, v in counts.items()}
        for label in CLASS_LABELS:
            if label not in set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().astype(int)):
                warnings.append(f"Class {label} ({class_name(label)}) is absent from the dataset.")

    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    numeric_invalid: dict[str, int] = {}
    for col in features:
        series = pd.to_numeric(df[col], errors="coerce")
        inf_count = int(np.isinf(series.to_numpy(dtype=float, copy=True)).sum()) if series.notna().any() else 0
        numeric_invalid[col] = inf_count
        if inf_count:
            warnings.append(f"Column {col} contains {inf_count} infinite value(s).")

    n_duplicates = int(df.duplicated().sum())
    feature_duplicates = int(df[features].duplicated().sum()) if features else 0
    if n_duplicates:
        warnings.append(f"{n_duplicates} fully duplicate records were found. They were not removed.")
    if feature_duplicates:
        warnings.append(
            f"{feature_duplicates} records share identical measurement-feature vectors. "
            "This can inflate apparent performance if those rows straddle the split."
        )

    constant_features = []
    near_constant = []
    for col in features:
        nunique = int(df[col].nunique(dropna=True))
        if nunique <= 1:
            constant_features.append(col)
        elif nunique <= 2:
            near_constant.append(col)
    if constant_features:
        warnings.append(f"Constant measurement features: {constant_features}")
    if near_constant:
        warnings.append(
            f"Near-constant measurement features (very few unique values): {near_constant}. "
            "LostPackets is stored from FlowMonitor and can under-count relative to Tx-Rx."
        )

    leakage_present = [c for c in IDENTIFIER_OR_LEAKAGE_COLUMNS if c in df.columns]
    if "DestinationPort" in df.columns:
        warnings.append(
            "DestinationPort is excluded from features because Sybil flows use ports 6001-6005 "
            "and hop traffic uses port 7000. Including it would leak scenario labels."
        )

    seed_counts = {}
    if SPLIT_GROUP_COLUMN in df.columns:
        seed_counts = {
            str(k): int(v) for k, v in df[SPLIT_GROUP_COLUMN].value_counts(dropna=False).items()
        }
        missing_seed = int(df[SPLIT_GROUP_COLUMN].isna().sum())
        if missing_seed:
            warnings.append(
                f"{missing_seed} row(s) have missing {SPLIT_GROUP_COLUMN} (legacy RunID=0 XML). "
                "Those rows are excluded from the seed-wise train/test split."
            )

    if "RunID" in df.columns and TARGET_COLUMN in df.columns:
        run_class_nunique = df.groupby("RunID")[TARGET_COLUMN].nunique()
        single_class_runs = int((run_class_nunique == 1).sum())
        if single_class_runs:
            warnings.append(
                f"{single_class_runs} RunID value(s) contain only one attack class. "
                "A random row-wise split would leak the same simulation run into train and test. "
                "Training uses a RandomSeed-wise split instead."
            )

    n_missing_delay = int(df["AverageDelayMs"].isna().sum()) if "AverageDelayMs" in df.columns else 0
    if n_missing_delay:
        warnings.append(
            f"{n_missing_delay} rows have missing AverageDelayMs/AverageJitterMs/ThroughputKbps. "
            "These measurements are empty when RxPackets=0 and are not filled with zeros. "
            "The training pipeline imputes medians fitted on training data only and adds missingness indicators."
        )

    warnings.append(
        "AttackType is a scenario-level label for the NS-3 run that produced the flow, "
        "not a per-packet attacker flag. Many flows in a DoS/Sybil/Blackhole run are ordinary UDP flows."
    )
    warnings.append(
        "The dataset contains simulated FlowMonitor observations (741 flows from 500-vehicle runs). "
        "It is historical simulation data, not live VANET traffic."
    )

    return {
        "dataset_path": str(DATASET_PATH),
        "n_records": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "class_counts": class_counts,
        "missing_values": missing,
        "infinite_values": numeric_invalid,
        "duplicate_records": n_duplicates,
        "duplicate_feature_vectors": feature_duplicates,
        "constant_features": constant_features,
        "near_constant_features": near_constant,
        "features_used": features,
        "excluded_columns": leakage_present,
        "seed_counts": seed_counts,
        "warnings": warnings,
        "errors": errors,
    }


def prepare_ml_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    report = build_validation_report(df)
    if report["errors"]:
        raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))

    features = report["features_used"]
    work = sanitize_numeric_frame(df, features + [TARGET_COLUMN, SPLIT_GROUP_COLUMN, "RunID"])
    work[TARGET_COLUMN] = work[TARGET_COLUMN].astype(int)

    excluded_legacy = work[SPLIT_GROUP_COLUMN].isna()
    n_excluded = int(excluded_legacy.sum())
    model_df = work.loc[~excluded_legacy].copy()
    model_df[SPLIT_GROUP_COLUMN] = model_df[SPLIT_GROUP_COLUMN].astype(float)

    split_info = {
        "method": "seed_wise_holdout",
        "group_column": SPLIT_GROUP_COLUMN,
        "train_seeds": TRAIN_SEEDS,
        "test_seeds": TEST_SEEDS,
        "excluded_legacy_rows": n_excluded,
        "reason": (
            "Each named simulation campaign uses a RandomSeed. Holding out seed 22 keeps "
            "entire runs out of training. A random stratified row split would place flows "
            "from the same XML into both sets."
        ),
    }
    return model_df, features, {"validation": report, "split": split_info}
