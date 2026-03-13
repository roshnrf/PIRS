# PIRS V2 — Predictive Insider Risk & Stabilization System

Pre-incident detection framework: detects behavioral drift **7-14 days before** a threat event.

## Folder Structure

```
pirs_v2/
  config.py              # Central config (CERT + LANL + model params)
  pipeline_cert.py       # Run full CERT pipeline
  pipeline_lanl.py       # Run full LANL pipeline
  extractors/
    cert_extractor.py    # CERT raw logs → features (HTTP enabled)
    lanl_extractor.py    # LANL auth.txt + proc.txt → features
  core/
    layer_1_baseline.py  # Personal behavioral baseline (per-user mean/std)
    layer_2_deviation.py # Daily z-score deviation from personal baseline
    layer_3_drift.py     # 14-day rolling drift slope detection
    layer_4_anomaly.py   # Ensemble: IsoForest + LSTM + OC-SVM on deviations
    layer_5_prediction.py# 7-day and 14-day breach trajectory prediction
    layer_6_personality.py# Personality context: COMPLIANT/SOCIAL/CAREFULL/RISK_TAKER/AUTONOMOUS
    layer_7_intervention.py# Personality-matched intervention selection (7 levels)
    layer_8_rl.py        # Q-learning intervention policy optimization
    layer_9_metrics.py   # EPR, PQ, SHAP explainability, cost savings
  validation/
    cert_validator.py    # Validate against 5 CERT insider users
    lanl_validator.py    # Validate against LANL redteam.txt
  outputs/
    cert/                # All CERT pipeline outputs
    lanl/                # All LANL pipeline outputs
```

## Dataset Placement

**CERT r6.2** — already in place:
```
pirs_backend/dataset/   (logon.csv, device.csv, file.csv, email.csv, http.csv, ...)
```

**LANL** — place files here:
```
PIRS/lanl_data/
  auth.txt
  proc.txt
  redteam.txt
```

## How to Run

### CERT (full run):
```bash
cd pirs_v2
python pipeline_cert.py
```

### CERT (resume from layer 3):
```bash
python pipeline_cert.py --from 3
```

### LANL (after data is downloaded):
```bash
python pipeline_lanl.py
```

### Validation:
```bash
python validation/cert_validator.py
python validation/lanl_validator.py
```

## Research Questions Answered

| RQ | Layer | Metric |
|----|-------|--------|
| RQ1: 7-14 day prediction | Layer 5 + cert_validator | EPR at 7d and 14d windows |
| RQ2: Personality-aware interventions | Layer 6+7 | PQ: matched vs generic |
| RQ3: Explainability | Layer 9 | SHAP top features per user |
| RQ4: Prevention quantification | Layer 9 | Cost savings estimate |
