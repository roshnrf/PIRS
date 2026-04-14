# PIRS — Predictive Intervention & Risk Stabilization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8973-brightgreen)
![EPR](https://img.shields.io/badge/EPR-59.75%25-brightgreen)
![PIMS](https://img.shields.io/badge/PIMS-1.18-brightgreen)
![Datasets](https://img.shields.io/badge/Validated%20On-CERT%20r6.2%20%7C%20LANL-blueviolet)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://pirs-v2.streamlit.app/)

> **Capstone Project — VIT Chennai, April 2026**
> Reem Fariha · Roshan A Rauof

---

## The Problem

Most security tools fire an alert **after** the data has been stolen, **after** the sabotage has happened. By the time the SIEM triggers, the damage is done.

Insider threats cost organizations an average of **$17.4 million per incident** (Ponemon & DTEX, 2023). The challenge is not just detecting them — it is **stopping them before they act**.

**PIRS answers a different question than every existing system:**

> Not *"Did we detect the insider?"*
> But *"Did we stop them?"*

---

## What PIRS Does

PIRS is a **9-layer behavioral AI pipeline** that:

1. Tracks how every user's behavior **drifts** from their personal baseline over time
2. Forecasts **7 days ahead** whether a user will breach a risk threshold
3. Profiles each user's **OCEAN Big Five personality** archetype
4. Selects a **personalized intervention** matched to their psychology
5. Uses **Q-learning** to optimize which intervention works best over time
6. Measures **prevention outcomes** — not just detection accuracy

---

## Architecture

![PIRS 9-Layer Pipeline Architecture](assets/architecture.jpg)

| Layers | Role |
|---|---|
| **L1–L4** | Early warning — ingest logs, build baseline, detect anomalies, forecast drift |
| **L5–L7** | Personalized response — OCEAN profiling, risk scoring, intervention selection |
| **L8** | Optimization — Q-learning refines intervention per user over time |
| **L9** | Measurement — EPR, PQ, PIMS, IES, TTC + ROC-AUC validation |

---

## Feature Engineering — 40 Semantic Features

Rather than using all 873 raw CERT columns, PIRS engineers **40 semantically meaningful features** mapped directly to the 5 insider threat scenarios:

| Group | Count | Key Signals |
|---|---|---|
| **After-hours timing** | 6 | after_hours_usb, after_hours_logon, after_hours_email |
| **USB exfiltration** | 5 | files_to_usb, docs_to_usb, compressed_to_usb |
| **Email exfiltration** | 8 | external_email_count, external_bcc_count, sent_external_ratio |
| **Web violations** | 8 | hack_site_visits, cloud_storage_visits, job_search_visits, leak_site_visits |
| **Activity patterns** | 5 | work_hour_ratio, total_logons, total_file_ops |
| **Composite scores** | 4 | exfiltration_score, policy_violation_score, insider_risk_composite |

**Composite exfiltration score:**
```python
exfiltration_score = (files_to_usb × 3.0) + (after_hours_usb × 2.0)
                   + (external_emails × 1.5) + (leak_site_visits × 2.5)
                   + (cloud_storage_visits × 2.0)
```

**Feature → Scenario mapping:**

| User | Scenario | Detected By |
|---|---|---|
| ACM2278 | Cloud upload / exfiltration | cloud_storage_visits, leak_site_visits |
| CMP2946 | Malicious download | hack_site_visits, after_hours_http |
| PLJ1771 | Espionage (recruited spy) | external_email_count, files_to_usb |
| CDE1846 | IP Theft before resignation | files_to_usb, docs_to_usb, after_hours_usb |
| MBG3183 | Sabotage (disgruntled) | hack_site_visits, timing_anomaly_score |

---

## OCEAN Personality Profiling

PIRS maps each user's Big Five OCEAN scores (ground-truth psychometric data in CERT r6.2, normalized from 10–50 to 0–1) to one of 5 behavioral archetypes:

| Archetype | Formula | Psychology | Optimal Intervention |
|---|---|---|---|
| **COMPLIANT** | (C + A) / 2 | Organized, rule-following | L2–L3: Policy reminders |
| **SOCIAL** | (E + A) / 2 | Outgoing, communicative | L4–L5: Peer/manager dialogue |
| **CAREFULL** | (C + (1−N)) / 2 | Disciplined, emotionally stable | L2–L3: Detailed risk reports |
| **RISK_TAKER** | (N + (1−C) + (1−A)) / 3 | Impulsive, boundary-pushing | L5–L6: Strong deterrents |
| **AUTONOMOUS** | (O + (1−E)) / 2 | Independent, self-directed | L3–L4: Autonomy-preserving nudges |

Primary archetype = `argmax` across 5 dimension scores, aggregated weekly.

---

## Results

### CERT r6.2 — 4,000 users · 515 days · 5 confirmed insider threats

| Metric | Result | Target | Status |
|---|---|---|---|
| **ROC-AUC** | **0.8973** | > 0.80 | Passed |
| **EPR** (Escalation Prevention Rate) | **59.75%** | 40–55% | Passed — above target |
| **PQ** (Preventability Quotient) | **0.5975** | 0.50–0.70 | Passed |
| **PIMS** (Personality-Intervention Match Score) | **1.18** | 1.15–1.30 | Passed |
| **TTC** (Time-to-Correction) | **47.8 hours** | 24–48h | Passed |
| False positive rate | **3.6%** of population flagged | — | Low |

### Per-Insider Detection

| User | Type | Labeled Days | In Pipeline Output | Top-5% Risk Rank |
|---|---|---|---|---|
| CDE1846 | IP Theft (pre-resignation) | 45 | Detected | Ranked top 5% |
| CMP2946 | Malicious Download | 20 | Detected | Ranked top 5% |
| ACM2278 | Cloud Exfiltration | 6 | Detected | Ranked top 5% |
| PLJ1771 | Espionage (1-day event) | 1 | Detected | Not in top 5% |
| MBG3183 | Sabotage (1-day event) | 1 | Detected | Not in top 5% |

> All 5 insiders present in `pirs_complete.csv` (1,361 insider-labeled rows out of 1,394,010 total).
> PLJ1771 and MBG3183 are single-day events — drift analysis requires ≥3 days of rising trend. Single-day impulsive events are a documented limitation of trajectory-based detection.

---

### Prevention Metrics Summary

![Prevention Metrics Summary](results/chart8_metrics_summary.png)

---

### ROC Curve — Insider Detection

![ROC Curve](results/chart6_roc.png)

> User-level ROC-AUC of **0.8973** — for any (insider, normal) user pair, PIRS ranks the insider higher 89.73% of the time.

---

### Risk Trajectories — Insider vs Normal Users

![Risk Trajectories](results/chart1_risk_trajectories.png)

> Shows how insider users (CDE1846 — IP Theft, 45 days) exhibit a clear upward drift in composite risk score in the days leading up to their malicious activity, while normal users remain stable.

---

### Early Warning Analysis

![Early Warning](results/chart2_early_warning.png)

> Drift-based breach warnings issued **before** the first labeled malicious day for gradual escalators — the core prevention capability of PIRS.

---

### Personality Archetype Distribution

![Personality Distribution](results/chart3_personality.png)

> OCEAN Big Five mapping across all 4,000 users — showing the distribution of COMPLIANT, SOCIAL, CAREFULL, RISK_TAKER, and AUTONOMOUS archetypes.

---

### Intervention Distribution by Archetype

![Interventions](results/chart5_interventions.png)

> How the 7 intervention levels are distributed across personality archetypes — RISK_TAKER users receive higher-level interventions; COMPLIANT users receive softer nudges.

---

### Per-Insider Risk Rankings

![Rankings](results/chart7_rankings.png)

> Risk ranking of all 5 insider users within the full 4,000-user population.

---

### LANL Cross-Dataset Validation — no retraining

| | |
|---|---|
| Dataset | Los Alamos National Laboratory (auth.txt 69 GB + proc.txt 15 GB) |
| Coverage | 12,416 users · 58 days · 97 red-team users (ground truth) |
| **ROC-AUC (user-level)** | **0.7429** |
| **Top-5% detection** | **20 / 97 red-team users (20.6%)** |

Same pipeline, different organization, different log schema, no retraining — confirms generalizability.

---

## Novel Prevention Metrics

No equivalent to these 5 metrics exists in prior insider threat literature:

| Metric | Formula | What It Answers |
|---|---|---|
| **EPR** | `100 × prevented / at_risk` | Did the intervention reduce the risk tier? |
| **PQ** | `prevented / at_risk` | What fraction of detected threats are preventable? |
| **PIMS** | `matched_rate / random_rate` | Does personality matching outperform generic response? |
| **IES** | `prevention_rate / (avg_level × TTC_days)` | How efficient is the intervention? |
| **TTC** | Weighted avg: L1=48h → L7=6h | How fast is risk contained? |

**PIMS = 1.18** means personality-matched interventions prevented **18% more escalations** than random (non-personalized) interventions under identical conditions. Random baseline uses mismatch penalty = 0.65.

---

## Comparison with Existing Literature

| System | Detection | Drift | Personality | Personalized Intervention | Prevention Metrics |
|---|---|---|---|---|---|
| Alzaabi & Mehmood, 2024 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Bin Sarhan & Altwaijry, 2023 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Villarreal-Vasquez et al., 2023 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Kotb DS-IID, 2024 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Nikiforova et al., 2024 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Shikonde & Nkongolo, 2025 | ✅ | Partial | ❌ | ❌ | ❌ |
| Ye et al. Federated, 2025 | ✅ | ❌ | ❌ | ❌ | ❌ |
| **PIRS (2026)** | ✅ | ✅ | ✅ | ✅ | ✅ |

Every prior system answers: *"Did we detect it?"* — PIRS answers: *"Did we stop it?"*

---

## Repository Structure

```
PIRS/
├── README.md
├── requirements.txt
├── results/
│   ├── chart1_risk_trajectories.png
│   ├── chart2_early_warning.png
│   ├── chart3_personality.png
│   ├── chart4_system_wide.png
│   ├── chart5_interventions.png
│   ├── chart6_roc.png
│   ├── chart7_rankings.png
│   ├── chart8_metrics_summary.png
│   └── risk_forecast.html
│
└── pirs_backend/
    ├── config.py                  ← All parameters in one place
    ├── data_extraction.py         ← Raw CERT log parser
    ├── feature_engineering.py     ← 873 raw → 40 semantic features
    ├── layer_1_3_baseline.py      ← Ensemble anomaly detection (IF+LSTM+SVM)
    ├── layer_4_drift.py           ← Behavioral drift + breach forecasting
    ├── layer_5_personality.py     ← OCEAN personality profiling
    ├── layer_6_interventions.py   ← Graduated intervention engine
    ├── layer_7_qlearning.py       ← Q-learning optimizer
    ├── layer_8_metrics.py         ← Prevention metrics (EPR/PQ/PIMS/IES/TTC)
    ├── layer_validation.py        ← ROC-AUC + insider detection evaluation
    ├── master_pipeline.py         ← Orchestrates all 9 layers end-to-end
    ├── lanl_extractor.py          ← LANL dataset feature extraction
    ├── lanl_pipeline.py           ← LANL cross-dataset validation
    └── pirs_dashboard_v2.py       ← Streamlit interactive dashboard
```

---

## Live Demo

**[https://pirs-v2.streamlit.app/](https://pirs-v2.streamlit.app/)** — Interactive dashboard showing risk trajectories, personality archetypes, intervention assignments, and prevention metrics across all 4,000 users.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/roshnrf/PIRS.git
cd PIRS/pirs_backend
pip install -r requirements.txt

# 2. Place datasets (see Dataset section below)

# 3. Run the full pipeline
#    Completed layers are skipped automatically (checkpointed)
python master_pipeline.py

# 4. Launch the dashboard locally
streamlit run pirs_dashboard_v2.py
```

**To re-run a specific layer**, delete its output file from `pirs_outputs/`:
```bash
# Example: re-run drift detection only
rm pirs_outputs/layer_4_drift.csv
python master_pipeline.py
```

---

## Key Configuration (config.py)

```python
# Ensemble model weights
ISOLATION_FOREST_WEIGHT = 0.50
LSTM_AUTOENCODER_WEIGHT = 0.35
OCSVM_WEIGHT            = 0.15

# Isolation Forest
ISOLATION_N_ESTIMATORS  = 100
ISOLATION_CONTAMINATION = 0.005   # 0.5% expected anomaly rate

# LSTM Autoencoder
LSTM_LATENT_DIM         = 32      # encoder bottleneck
LSTM_EPOCHS             = 10
LSTM_BATCH_SIZE         = 64

# One-Class SVM
OCSVM_NU                = 0.005   # outlier fraction
OCSVM_KERNEL            = 'rbf'

# Drift detection
DRIFT_WINDOW            = 7       # days lookback for regression
FORECAST_HORIZON        = 7       # days forward projection

# Q-learning
Q_LEARNING_ALPHA        = 0.1     # learning rate
Q_LEARNING_GAMMA        = 0.6     # discount factor
Q_LEARNING_EPSILON      = 0.2     # exploration rate (ε-greedy)
Q_LEARNING_EPISODES     = 100

# Prevention metrics
MISMATCH_PENALTY        = 0.65    # random interventions are 35% less effective
```

---

## Dataset: CERT r6.2

| Property | Value |
|---|---|
| Producer | Software Engineering Institute, Carnegie Mellon University |
| Users | 4,000 simulated employees |
| Days | 515 |
| Raw records | 1,393,810 rows × 893 columns (4.2 GB) |
| Activity domains | logon (12), USB (18), file (333), email (231), HTTP (249) |
| OCEAN scores | Included per user — scale 10–50 |
| Insider scenarios | 5 labeled users with ground-truth malicious day ranges |
| Citation | Glasser & Lindauer, IEEE S&P Workshops, 2013 |
| Access | [CMU KiltHub](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) |

Place downloaded files in `pirs_backend/dataset/`.

---

## Standards & Alignment

| Standard | How PIRS Aligns |
|---|---|
| **NIST SP 800-94** | 7-level intervention ladder follows NIST graduated response guidelines |
| **IEEE ML standards** | Ensemble design, cross-dataset validation, and reporting follow IEEE best practices |
| **NEO-PI-R / Big Five** | 50-year validated psychometric model (Costa & McCrae, 1992) |
| **SDG 16** | Peace, Justice and Strong Institutions — data security protects organizational integrity |

---

## Future Work

- **Real-time streaming** — Kafka/Flink integration for live log processing
- **SHAP explainability** — Human-readable explanation for every flag
- **Graph Neural Networks** — Detect coordinated multi-user insider collusion
- **Federated learning** — Cross-organization model sharing without raw log exposure
- **LLM behavioral profiling** — Nuanced signals from email text content
- **Dynamic OCEAN re-scoring** — Personality archetypes updated in real time

---

## Citation

```bibtex
@misc{pirs2026,
  title   = {PIRS: Predictive Intervention and Risk Stabilization for Insider Threat Prevention},
  author  = {Fariha, Reem and Rauof, Roshan A},
  year    = {2026},
  school  = {VIT Chennai},
  note    = {MS Capstone Project, School of Computer Science and Engineering}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
