# PIRS V2 Dashboard — Streamlit Cloud Deployment

## What's been prepared

| File/Folder | Size | Purpose |
|---|---|---|
| `pirs_backend/deploy_data/` | **386 KB total** | Pre-computed summary data (replaces 501 MB pirs_complete.csv) |
| `pirs_backend/requirements_dashboard.txt` | — | Minimal deps for cloud (streamlit, pandas, plotly) |
| `pirs_backend/.streamlit/config.toml` | — | Dark theme + server settings |
| `pirs_backend/pirs_dashboard_v2.py` | — | Updated to use deploy_data/ when full CSV not available |

### deploy_data/ files (386 KB total)
- `dashboard_user_summary.csv` — 4,000 users × peak/mean/last risk (251 KB)
- `dashboard_insider_trajectories.csv` — 5 insiders + 3 normal users, day-by-day (118 KB)
- `dashboard_daily_flags.csv` — 516 days × system-wide stats (17 KB)
- `dashboard_metrics.csv` — scalar metrics (EPR, PQ, ROC-AUC, etc.)
- `dashboard_risk_distribution.csv` — histogram bins
- `dashboard_personality_dist.csv` — personality type counts
- `dashboard_intervention_dist.csv` — intervention level counts
- `dashboard_lanl_summary.csv` — LANL validation results

---

## Step-by-step deployment

### 1. Commit everything to GitHub

```bash
cd C:\Users\rosha\Documents\PIRS

# Stage the dashboard + deploy data + config
git add pirs_backend/pirs_dashboard_v2.py
git add pirs_backend/deploy_data/
git add pirs_backend/requirements_dashboard.txt
git add pirs_backend/.streamlit/config.toml
git add pirs_backend/prepare_deploy_data.py
git add DEPLOY.md

git commit -m "Add deploy_data summaries + Streamlit Cloud config for hosted dashboard"
git push
```

### 2. Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Set:
   - **Repository**: `<your-github-username>/PIRS`
   - **Branch**: `main` (or your branch)
   - **Main file path**: `pirs_backend/pirs_dashboard_v2.py`
   - **Advanced settings → Requirements file**: `pirs_backend/requirements_dashboard.txt`
4. Click **Deploy** — it'll be live in ~2 minutes

### 3. Verify it works

The dashboard will load in cloud mode:
- ✅ Overview tab: gauge + histogram from deploy_data
- ✅ Insider Analysis: 5 insider trajectory charts from deploy_data
- ✅ Risk Monitor: all 4,000 users searchable from user_summary
- ✅ LANL tab: all static (no CSV needed)
- ✅ Pipeline, Interventions, Applications: all static content

---

## Regenerating deploy_data (after re-running pipeline locally)

```bash
cd pirs_backend
python prepare_deploy_data.py
git add deploy_data/
git commit -m "Update deploy_data with latest pipeline results"
git push
```
Streamlit Cloud auto-redeploys on push.

---

## Local development (full data)

Nothing changes locally — `pirs_complete.csv` is still used when present.

```bash
cd pirs_backend
streamlit run pirs_dashboard_v2.py --server.port 8502
```
