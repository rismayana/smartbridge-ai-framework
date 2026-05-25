# Weighted Ensemble Framework for Structural Health Monitoring (SHM)

> AI-driven anomaly detection framework for cable-stayed bridge monitoring using multi-sensor fusion, temporal modeling, statistical outlier analysis, and graph-based spatial intelligence.

> **Status**: In Progress  
> **Target Publication**: Journal of Computing Theories and Applications (JCTA) — August 2026

---

# Overview

This research develops an AI-based anomaly detection framework for Structural Health Monitoring Systems (SHMS) on long-span cable-stayed bridges.

Conventional SHMS deployments commonly rely on static threshold-based monitoring, which often suffers from:

- high false positive rates,
- inability to capture inter-sensor relationships,
- limited adaptability to operational/environmental variability,
- poor scalability for complex multi-channel systems.

To address these limitations, this project proposes a weighted ensemble framework combining:

| Model | Primary Function |
|---|---|
| LSTM Autoencoder | Temporal anomaly detection |
| Isolation Forest | Statistical multi-channel outlier detection |
| Graph Neural Network (GNN) | Spatial correlation anomaly detection |

The framework is designed for real-world infrastructure monitoring scenarios where sensor reliability, environmental variability, and operational robustness are critical considerations.

---

# Industrial Relevance

This project is aligned with emerging smart infrastructure and infrastructure resilience initiatives, including:

- Smart Bridge Monitoring
- Predictive Maintenance
- Infrastructure Intelligence
- AI-Assisted Structural Diagnostics
- Digital Twin Ecosystems
- Intelligent Transportation Infrastructure
- Real-Time Infrastructure Monitoring

---

# System Architecture

```text
Sensor Layer
(Accelerometer, Cable Tensionmeter, Environmental Sensors)
        ↓
Data Acquisition System (DAQ)
        ↓
Preprocessing & Signal Validation
        ↓
Feature Engineering & Correlation Analysis
        ↓
AI Models
├── LSTM Autoencoder
├── Isolation Forest
└── Graph Neural Network
        ↓
Weighted Ensemble Fusion
        ↓
Anomaly Scoring & Classification
        ↓
Monitoring / Alert Layer
```

---

# Real-World Engineering Challenges

This project addresses several operational SHMS challenges commonly encountered in real infrastructure systems:

- Sensor drift and calibration offset
- Flatline / inactive sensor detection
- Environmental variability
- Missing or corrupted sensor data
- Multi-channel correlation complexity
- False positive reduction
- Spatial-temporal anomaly interaction
- Real-time inference constraints

Unlike benchmark-only AI studies, this research is based on operational monitoring data with heterogeneous sensor behavior and imperfect field conditions.

---

# Dataset Summary

| Component | Description |
|---|---|
| Infrastructure Type | Cable-Stayed Bridge |
| Monitoring System | Structural Health Monitoring System (SHMS) |
| Sensor Channels | Selected subset from 90 total channels |
| Sensor Types | Accelerometer, Cable Tensionmeter, Temperature, Wind |
| Sampling Rate | 100 Hz (environmental sensor: 1 Hz) |
| Data Characteristics | Real operational infrastructure monitoring data |

---

# Data Availability

Due to infrastructure security and institutional confidentiality agreements, raw SHMS datasets are not publicly distributed.

This repository focuses on:

- methodology,
- pipeline architecture,
- preprocessing workflow,
- AI framework implementation,
- and reproducible engineering structure.

Synthetic or simplified demonstration datasets may be released in future versions.

---

# Repository Structure

```text
shms-weighted-ensemble-framework/
│
├── 01_paper/          # manuscript, figures, submission materials
├── 02_data/           # raw, processed, abnormal, statistics
├── 03_code/           # Python pipeline + notebooks
├── 04_models/         # trained model checkpoints
├── 05_results/        # metrics, plots, evaluation outputs
├── 06_references/     # cited literature
├── assets/            # architecture diagrams, figures
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Pipeline Workflow

| Phase | Module | Description |
|---|---|---|
| 1 | Data Loader | Loading, integrity check, exploratory analysis |
| 2 | Preprocessing | Normalization, windowing, correlation analysis |
| 3A | LSTM Autoencoder | Temporal anomaly modeling |
| 3B | Isolation Forest | Statistical anomaly detection |
| 3C | Graph Neural Network | Spatial anomaly modeling |
| 4 | Ensemble Fusion | Weighted anomaly fusion |
| 5 | Evaluation | Metrics, visualization, ablation analysis |

---

# Installation

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare dataset

Place local CSV files inside:

```text
02_data/raw/
02_data/abnormal/
```

## 3. Run pipeline

```bash
cd 03_code
python run_pipeline.py
```

Or run interactively using Jupyter Notebook:

```bash
jupyter notebook 03_code/notebooks/
```

---

# Configuration

All project configurations are centralized in:

```text
03_code/shms_config.py
```

This includes:

- sensor channel selection,
- file paths,
- preprocessing parameters,
- model hyperparameters,
- ensemble weighting configuration.

---

# Experimental Focus

Current experimental objectives include:

- Temporal anomaly detection
- Spatial correlation anomaly detection
- Multi-model ensemble robustness
- False positive reduction
- Sensor reliability analysis
- Ablation studies
- Infrastructure-oriented evaluation metrics

---

# Future Development

Planned future extensions include:

- Explainable AI (XAI) integration
- Edge AI deployment
- Online learning framework
- Real-time SHMS dashboard
- Digital Twin integration
- Multi-bridge transfer learning
- Infrastructure risk scoring system

---

# Research Context

This project is part of ongoing research in:

- Structural Health Monitoring (SHM)
- Infrastructure AI
- Smart Infrastructure Systems
- Edge Intelligence
- Multi-Sensor AI Analytics

---

# Disclaimer

This repository is intended for research and engineering demonstration purposes only.

Some implementation details, infrastructure identifiers, and datasets have been partially abstracted or anonymized due to confidentiality and infrastructure security considerations.

---

# Contact

Research Area:

- Structural Health Monitoring (SHM)
- Infrastructure AI
- Smart Infrastructure Systems
- Computer Vision for Infrastructure
- Edge AI Monitoring Systems
