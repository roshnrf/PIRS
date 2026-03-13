# PIRS - Predictive Insider Risk & Stabilization System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8554-brightgreen)
![Early Warning](https://img.shields.io/badge/Early%20Warning-3%2F5%20insiders-brightgreen)
![Datasets](https://img.shields.io/badge/Validated%20On-CERT%20%7C%20LANL-blueviolet)

---

## Most security tools catch threats too late.

By the time an alert fires, the data is already gone, or the damage is done.

**PIRS takes a different approach: it watches how people *change* over time, and predicts a threat 3 to 14 days before it happens.**

When someone is about to go rogue, their behaviour shifts. They start copying files at odd hours, visiting job boards, sending more external emails. PIRS tracks these patterns for every individual user, spots the drift, and flags the risk early enough to actually do something about it.

---

## What makes it different?

| | Traditional SIEM | **PIRS** |
|--|----------------|---------|
| When it alerts | After the attack | **Up to 14 days before** |
| How it models users | Same rules for everyone | **Personal baseline per user** |
| What it does with the alert | Sends a generic notification | **Matches a response to the person's personality** |
| Does it learn? | No | **Yes, Q-learning optimises the response over time** |
| Can you trust the output? | Black box | **SHAP explains exactly why each user was flagged** |

---

## How it works — 9 layers

Think of it as a pipeline that goes from raw logs to a prevention action:

```
Raw logs  →  Who is this person normally?       (Layer 1: Personal Baseline)
          →  How different is today?             (Layer 2: Daily Deviation)
          →  Is the change accelerating?         (Layer 3: Drift Detection)
          →  Is something genuinely wrong?       (Layer 4: Anomaly Ensemble)
          →  When will this peak?                (Layer 5: Breach Prediction) ← core
          →  What kind of person are they?       (Layer 6: Personality Profile)
          →  What's the right intervention?      (Layer 7: Intervention Match)
          →  What's worked before?               (Layer 8: Q-Learning Policy)
          →  Did we actually prevent anything?   (Layer 9: Prevention Metrics)
```

Layer 5 is where prediction happens. Layers 6-9 are about *what to do with it*, because flagging a threat means nothing if the response makes things worse.

---

## Results

### Tested on CERT r6.2 — 4,000 employees, 515 days, 5 real insider threats

```
  ROC-AUC          0.8554   ✓  (target ≥ 0.75)
  Early Warning    3 / 5    ✓  insiders flagged 3 days before their attack
  Prevention Rate  40.0%    ✓  (target ≥ 40%)
  Prevention Quality  0.93  ✓  personality-matched vs generic responses
```

How each insider looked 3 days before their attack:

| Who | What they did | Risk score (3d before) | Caught? |
|-----|--------------|----------------------|---------|
| ACM2278 | Uploaded files to Wikileaks | **8.47 / 10** (CRITICAL) | ✅ Yes |
| CMP2946 | Copied data to USB drive | **6.41 / 10** (HIGH) | ✅ Yes |
| CDE1846 | Exfiltrated data via email | **4.13 / 10** (ELEVATED) | ✅ Yes |
| PLJ1771 | Installed a keylogger | 0.56 / 10 (normal) | ❌ Missed |
| MBG3183 | Uploaded to Dropbox | 1.54 / 10 (normal) | ❌ Missed |

3 out of 5 caught before the attack. The 2 missed cases showed almost no behavioural change beforehand, a known hard case in insider threat research.

### Validated on LANL — 12,416 users, 58 days, 97 labelled attackers

```
  ROC-AUC (user level)   0.7429   ✓
  Top 5% detection       20 / 97 attackers surfaced in top 5% risk
  Top 10% detection      ~40 / 97 attackers surfaced in top 10% risk
```

Same pipeline, completely different dataset, different organisation, different log format. Still works. That's the point of cross-dataset validation.

---

## V1 → V2: Why we rebuilt it

V1 was built first as a detection baseline. It could spot anomalies but had no concept of *time* or *who the person is*. V2 is the full system.

| | V1 | V2 |
|--|----|----|
| Goal | Detect anomalies | **Predict breaches** |
| User model | Global rules | **Personal baseline** |
| Time awareness | None | **14-day rolling drift** |
| Prediction | ❌ | ✅ 7d + 14d forecast |
| Personality | ❌ | ✅ 5 profiles |
| Interventions | Generic alert | **7 levels, personality-matched** |
| RL | ❌ | ✅ Q-learning |
| Explainability | ❌ | ✅ SHAP per user |
| ROC-AUC | ~0.72 | **0.8554** |
| Early warning | 1 / 5 | **3 / 5** |

---

## What's in this repo

```
pirs_v2/
├── pipeline_cert.py        ← run CERT end-to-end
├── pipeline_lanl.py        ← run LANL end-to-end
├── config.py               ← all settings in one place
├── extractors/
│   ├── cert_extractor.py   ← turns raw CERT logs into 873 features
│   └── lanl_extractor.py   ← handles the 69GB LANL auth file in chunks
├── core/
│   ├── layer_1_baseline.py through layer_9_metrics.py
├── validation/
│   ├── cert_validator.py   ← checks results against the 5 known insiders
│   └── lanl_validator.py   ← checks against LANL red-team labels
└── outputs/cert/           ← pre-computed results (CSVs)

results/
└── PIRS_Results.xlsx       ← full results table, both datasets
```

---

## Run it yourself

```bash
pip install -r requirements.txt

# CERT pipeline
cd pirs_v2
python pipeline_cert.py

# LANL pipeline
python pipeline_lanl.py

# Dashboard
cd ../pirs_backend
streamlit run pirs_dashboard.py

# Resume from a specific layer (completed layers are skipped automatically)
python pipeline_cert.py --from 4
```

**Datasets are not included** — they're large and require separate access:
- CERT r6.2: [CMU CERT Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
- LANL: [LANL Unified Host and Network](https://csr.lanl.gov/data/cyber1/)

Place them here once downloaded:
```
pirs_backend/dataset/      ← CERT files (logon.csv, email.csv, file.csv, etc.)
lanl_data/                 ← LANL files (auth.txt, proc.txt, redteam.txt)
```

---

## Prevention metrics

Five metrics were defined to measure whether the system actually *prevents* threats, not just flags them:

| Metric | What it asks |
|--------|-------------|
| **EPR** — Early Prevention Rate | Did we flag it before the attack? |
| **PQ** — Prevention Quality | Did personality-matching improve the response? |
| **PIMS** — Prevention Impact Score | How much did the risk actually drop? |
| **IES** — Intervention Effectiveness | Did the intervention work for this person? |
| **TTC** — Time to Contain | How long from first flag to risk stabilised? |

---

## MS Capstone — March 2026
