# VANET ML detection and Streamlit dashboard

Workspace: `/home/bindu/VANET_500_Experiment`

This folder adds Random Forest and SVM training, held-out evaluation, and a Streamlit dashboard **without modifying** NS-3 sources, original XML files, NetAnim traces, or `dataset/vanet_500_dataset.csv`.

Protected elsewhere (not touched): `/home/bindu/VANET_Project`, `/home/bindu/VANET_Project_BACKUP`.

## What was implemented

- `train_models.py` validates the CSV, trains Random Forest and SVM, evaluates both on the same held-out seed, and writes pipelines plus metrics.
- `app.py` is the SOC-themed dashboard (executive view, dataset explorer, attack detection form, model performance, about).
- `ml_core.py` holds shared paths, the feature policy, and dataset validation helpers.

## Dataset used

File: `dataset/vanet_500_dataset.csv`

- About **741** FlowMonitor **flow-level** records from NS-3.48 runs with **500 vehicle nodes**.
- Target: `AttackType` — 0 Normal, 1 DoS, 2 Sybil, 3 Blackhole.
- This is **historical simulation data**, not live VANET traffic.

### Features included

`TxPackets`, `RxPackets`, `LostPackets`, `TxBytes`, `RxBytes`, `PacketLossRate`, `AverageDelayMs`, `AverageJitterMs`, `ThroughputKbps`

### Features excluded (identifiers or leakage)

`FlowID`, `RunID`, `RandomSeed`, `Scenario`, `AttackType`, addresses, node IDs, `Protocol`, `SourcePort`, `DestinationPort`, `SourceXml`, `MeasurementLevel`

`DestinationPort` is excluded because Sybil uses 6001–6005 and hop traffic uses 7000.

### Missing measurements

Delay, jitter, and throughput are empty when `RxPackets = 0`. Training **does not fill those CSV cells with zeros**. The sklearn pipeline imputes **training medians** and adds **missingness indicators**.

## Split (leakage control)

Most `RunID` values contain only one attack class. A random row-wise split would put flows from the same XML into train and test.

Training therefore uses a **RandomSeed-wise holdout**:

- Train: seeds **1** and **11**
- Test: seed **22**
- Three legacy `RunID=0` rows with missing seed are excluded from the split

Both models are scored on that same test set.

## Commands (Ubuntu, project root)

```bash
cd /home/bindu/VANET_500_Experiment
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

If the venv is not already active:

```bash
cd /home/bindu/VANET_500_Experiment
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_models.py
.venv/bin/streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

If models are missing, the dashboard tells you to run `python train_models.py`. It does not invent accuracy numbers.

## Outputs (new; original XML kept)

Pipelines:

- `models/random_forest_pipeline.joblib`
- `models/svm_pipeline.joblib`
- `models/selected_pipeline.joblib`
- `models/selected_model_meta.json`

Evaluation is written under `results/ml/` so existing NetAnim/FlowMonitor XML files in `results/` are not overwritten:

- `results/ml/evaluation.json`
- `results/ml/metrics_summary.csv`
- `results/ml/model_comparison.csv`
- `results/ml/classification_report_random_forest.csv`
- `results/ml/classification_report_svm.csv`
- `results/ml/confusion_matrix_random_forest.png`
- `results/ml/confusion_matrix_svm.png`
- `results/ml/model_comparison.png`
- `results/ml/dataset_validation.json`

## Dashboard pages

1. Executive dashboard — record counts, class mix, trained models, real test metrics, warnings.
2. Dataset explorer — read-only CSV preview, types, filters, missingness, download.
3. Attack detection — form over the same features/pipelines; RF vote shares or SVM decision scores, labelled honestly.
4. Model performance — test metrics, confusion matrices, reports, methodology, limits.
5. About — VANET/ITS context and the actual Sybil/Blackhole simulation limits.

## Known limits (do not over-claim)

- Sybil is **application-level** extra UDP identities.
- Blackhole is an **application relay drop**, not a routing-protocol blackhole.
- Labels are **scenario-level**: ordinary flows in a DoS run are still labelled DoS.
- Test size is one extra seed (~246 flows). That is small for operational claims.
- Scores are **not calibrated probabilities** unless a calibration step is added later (it is not).
- The CSV alone does **not** establish real-world ITS detection performance.
