# Cyber Attack Detection in VANET for Connected Vehicles Using Machine Learning

## Overview

Vehicular Ad Hoc Networks (VANETs) enable communication between vehicles and roadside infrastructure to support Intelligent Transportation Systems (ITS). However, these networks are vulnerable to cyberattacks that can affect communication reliability, network performance, and vehicle safety.

This project presents a machine learning-based approach for detecting and classifying cyberattacks in VANET environments. A 500-vehicle network is simulated using NS-3 under normal traffic and different attack scenarios. The generated network traffic data is processed using Python and used to train and evaluate machine learning models.

The project integrates network simulation, dataset generation, machine learning classification, and an interactive Streamlit dashboard into a single workflow.

## Objectives

- Simulate a VANET environment containing 500 vehicles.
- Generate network traffic data under normal and attack conditions.
- Simulate and analyze different cyberattack scenarios.
- Extract and validate network traffic features using Python.
- Train machine learning models to classify network traffic.
- Compare Random Forest and Support Vector Machine (SVM) performance.
- Visualize dataset characteristics, attack classifications, and model evaluation results through a Streamlit dashboard.

## Attack Scenarios

The simulation includes four traffic classes:

| Class ID | Traffic Class | Description |
|---|---|---|
| 0 | Normal | Represents network traffic without an intentionally enabled attack. |
| 1 | DoS | Denial of Service scenario involving excessive traffic intended to affect network communication. |
| 2 | Sybil | Application-level simulation in which a vehicle sends traffic using multiple claimed identities. |
| 3 | Blackhole | Application-level relay dropping scenario in which the relay receives packets but does not forward them. |

**Note:** The Sybil and Blackhole scenarios are implemented at the application level. They do not represent complete routing-protocol implementations of these attacks.

## Technologies Used

| Technology | Purpose |
|---|---|
| NS-3 | Network simulation |
| NetAnim | Visualization of network activity |
| Python | Dataset processing and machine learning |
| Pandas | Dataset manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Random Forest | Traffic classification |
| Support Vector Machine (SVM) | Traffic classification |
| Streamlit | Interactive dashboard |
| Joblib | Saving and loading trained models |

## System Architecture

The project follows this workflow:

1. **VANET Simulation:** A 500-vehicle network is created using NS-3.
2. **Attack Simulation:** Normal, DoS, Sybil, and Blackhole scenarios are executed.
3. **Data Collection:** Network simulation outputs are generated, including FlowMonitor XML files and metadata.
4. **Dataset Generation:** Python scripts extract network traffic information and construct a structured dataset.
5. **Dataset Validation:** The generated dataset is checked for data quality and consistency.
6. **Model Training:** Random Forest and SVM classifiers are trained using the selected traffic features.
7. **Model Evaluation:** The classifiers are evaluated using held-out data and classification metrics.
8. **Dashboard Visualization:** Streamlit displays dataset insights, attack classification results, and model performance.

## Dataset

The project uses simulation-generated network traffic data.

The combined dataset is stored at:

`dataset/vanet_500_dataset.csv`

Depending on the files included in the repository, the dataset may contain features such as:

- Transmitted packets
- Received packets
- Lost packets
- Transmitted bytes
- Received bytes
- Packet loss rate
- Average delay
- Average jitter
- Throughput
- Traffic class label

The dataset contains flow-level records. Therefore, the number of dataset rows is not necessarily equal to the number of simulated vehicles.

The `AttackType` field identifies the traffic class and is used as the classification target rather than as an input feature.

## Machine Learning Models

### 1. Random Forest Classifier

Random Forest combines multiple decision trees to classify network traffic. It can model nonlinear relationships between network traffic characteristics and attack classes.

### 2. Support Vector Machine (SVM)

SVM identifies decision boundaries between traffic classes. Feature scaling is important when training SVM models.

### Model Evaluation

The models are evaluated using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Per-class classification metrics

The actual evaluation results should be obtained from the generated model reports. No performance values are assumed in this README.

## Streamlit Dashboard

The project includes an interactive dashboard for exploring the dataset and reviewing model predictions and evaluation results.

Depending on the implemented dashboard features, users can explore:

- Dataset characteristics and class distribution
- Attack detection results
- Random Forest and SVM performance
- Confusion matrices and classification reports
- Model comparison results

The dashboard uses the generated simulation dataset and trained models. It does not represent a live vehicle-network monitoring system.

## Project Information

**Project Type:** Final-Year Group Project

**Domain:** Machine Learning, Cybersecurity, and Intelligent Transportation Systems

### Project Contributions

The project involves NS-3-based VANET simulation, cyberattack scenario generation,dataset preparation, machine learning model development, and an interactive Streamlit dashboard.

Each team member contributed to the project as part of the collaborative development process.
## Team Contributions

This project was developed collaboratively as a final-year group project.

### My Contributions — Development and Implementation
- Implemented the 500-vehicle VANET simulation using NS-3.
- Developed and executed normal traffic and cyberattack scenarios, including DoS, Sybil, and Blackhole attacks.
- Generated, extracted, and validated simulation-based network traffic datasets using Python.
- Implemented and evaluated Random Forest and Support Vector Machine (SVM) classifiers.
- Developed the Streamlit dashboard for dataset exploration, attack classification, and model performance visualization.
- Integrated the simulation outputs, dataset processing, machine learning models, and dashboard into the project workflow.

### Teammate Contributions — Research and Documentation
- Conducted literature surveys and reviewed related research papers.
- Prepared project reports and technical documentation.
- Developed presentation slides for project reviews and demonstrations.
- Assisted with organizing research findings and documenting the project methodology.

Both team members collaborated on the overall project and its academic objectives.
