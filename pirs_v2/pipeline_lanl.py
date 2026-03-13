"""
PIRS V2 - LANL PIPELINE RUNNER
================================
Runs the PIRS core detection framework on the LANL dataset.

Layers run: 0 (extraction) → 1-5 (baseline, deviation, drift, anomaly, prediction)
            → 7 (intervention, no personality) → 9 (metrics)

Layers SKIPPED for LANL:
  Layer 6: Personality (no OCEAN data in LANL)

Validation: evaluated against redteam.txt ground truth.

Usage:
    python pipeline_lanl.py              # Full run
    python pipeline_lanl.py --from 3    # Resume from layer 3
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import LANLConfig, ModelConfig

cfg  = LANLConfig()
mcfg = ModelConfig()

from extractors.lanl_extractor  import run_lanl_extraction
from core.layer_1_baseline      import run as run_baseline, get_feature_cols
from core.layer_2_deviation     import run as run_deviation
from core.layer_3_drift         import run as run_drift
from core.layer_4_anomaly       import run as run_anomaly
from core.layer_5_prediction    import run as run_prediction
from core.layer_7_intervention  import run as run_intervention
from core.layer_8_rl            import run as run_rl
from core.layer_9_metrics       import run as run_metrics

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILES = {
    0: os.path.join(cfg.OUTPUT_DIR, 'lanl_features.csv'),
    3: os.path.join(cfg.OUTPUT_DIR, 'lanl_after_drift.csv'),
    4: os.path.join(cfg.OUTPUT_DIR, 'lanl_after_anomaly.csv'),
    5: os.path.join(cfg.OUTPUT_DIR, 'lanl_after_prediction.csv'),
    7: os.path.join(cfg.OUTPUT_DIR, 'lanl_after_intervention.csv'),
    9: os.path.join(cfg.OUTPUT_DIR, 'lanl_complete.csv'),
}

METRICS_FILE = os.path.join(cfg.OUTPUT_DIR, 'lanl_metrics.csv')


def checkpoint_exists(layer):
    return os.path.exists(CHECKPOINT_FILES.get(layer, ''))


def save(df, layer):
    path = CHECKPOINT_FILES.get(layer)
    if path:
        df.to_csv(path, index=False)
        print(f"  [SAVED] {os.path.basename(path)} ({len(df):,} rows)")


def load(layer):
    path = CHECKPOINT_FILES[layer]
    print(f"  [LOAD] {os.path.basename(path)}")
    return pd.read_csv(path, low_memory=False)


def build_redteam_malicious_days() -> dict:
    """Build per-user malicious day dict from redteam.txt for metrics."""
    from extractors.lanl_extractor import load_redteam, seconds_to_day, clean_user
    _, rt_df = load_redteam()
    malicious = {}
    for _, row in rt_df.iterrows():
        user = row['user']
        day  = int(row['day'])
        malicious.setdefault(user, []).append(day)
    return malicious


def run_pipeline(start_from: int = 0):
    print("=" * 65)
    print("  PIRS V2 -- LANL Pipeline")
    print("  Cross-dataset validation (red team attack detection)")
    print("=" * 65)

    if not os.path.exists(cfg.AUTH_FILE):
        print(f"\n[ERROR] auth.txt not found at: {cfg.AUTH_FILE}")
        print("  Please place LANL files in: C:\\Users\\rosha\\Documents\\PIRS\\lanl_data\\")
        return None

    t_start  = time.time()
    dev_cols = None

    # ----------------------------------------------------------------
    # LAYER 0: EXTRACTION
    # ----------------------------------------------------------------
    if start_from <= 0:
        if checkpoint_exists(0):
            print("\n[L0] Extraction checkpoint found -- loading...")
            df = load(0)
        else:
            df = run_lanl_extraction()
    else:
        for ck in sorted(CHECKPOINT_FILES.keys(), reverse=True):
            if ck <= start_from and checkpoint_exists(ck):
                df = load(ck)
                break

    # LANL label column is 'redteam' (not 'insider')
    label_col  = 'redteam'
    feature_cols = get_feature_cols(df, exclude=['user', 'day', 'redteam'])
    print(f"\n  Feature columns: {len(feature_cols)}")

    # ----------------------------------------------------------------
    # LAYERS 1-2: BASELINE + DEVIATION
    # ----------------------------------------------------------------
    if start_from <= 2:
        print("\n" + "-"*40)
        baseline, feature_cols = run_baseline(df, feature_cols)
        df, dev_cols = run_deviation(df, baseline, feature_cols)
    else:
        dev_cols = [c for c in df.columns if c.endswith('_dev')]

    # ----------------------------------------------------------------
    # LAYER 3: DRIFT
    # ----------------------------------------------------------------
    if start_from <= 3:
        print("\n" + "-"*40)
        df = run_drift(df)
        save(df, 3)
    elif start_from > 3:
        df = load(3)
        dev_cols = [c for c in df.columns if c.endswith('_dev')]

    # ----------------------------------------------------------------
    # LAYER 4: ANOMALY
    # ----------------------------------------------------------------
    if start_from <= 4:
        print("\n" + "-"*40)
        if dev_cols is None:
            dev_cols = [c for c in df.columns if c.endswith('_dev')]
        df = run_anomaly(df, dev_cols)
        save(df, 4)
    elif start_from > 4:
        df = load(4)
        dev_cols = [c for c in df.columns if c.endswith('_dev')]

    # ----------------------------------------------------------------
    # LAYER 5: PREDICTION
    # ----------------------------------------------------------------
    if start_from <= 5:
        print("\n" + "-"*40)
        df = run_prediction(df)
        save(df, 5)
    elif start_from > 5:
        df = load(5)
        dev_cols = [c for c in df.columns if c.endswith('_dev')]

    # ----------------------------------------------------------------
    # LAYER 7: INTERVENTION (no personality for LANL)
    # ----------------------------------------------------------------
    if start_from <= 7:
        print("\n" + "-"*40)
        df = run_intervention(df)    # Will use DEFAULT_RULES (no personality)
        save(df, 7)

    # ----------------------------------------------------------------
    # LAYER 8: Q-LEARNING
    # ----------------------------------------------------------------
    if start_from <= 8:
        print("\n" + "-"*40)
        # Rename label col for RL compatibility
        df['insider'] = df.get(label_col, 0)
        df, agent = run_rl(df)

    # ----------------------------------------------------------------
    # LAYER 9: METRICS
    # ----------------------------------------------------------------
    if start_from <= 9:
        print("\n" + "-"*40)
        if dev_cols is None:
            dev_cols = [c for c in df.columns if c.endswith('_dev')]

        # Build malicious days dict for LANL
        malicious_days = build_redteam_malicious_days()
        n_lanl_users = df['user'].nunique()

        metrics = run_metrics(
            df,
            dev_cols,
            malicious_days=malicious_days,
            n_users=n_lanl_users,
        )

        save(df, 9)
        metrics['summary_df'].to_csv(METRICS_FILE, index=False)
        print(f"  [SAVED] {os.path.basename(METRICS_FILE)}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  LANL Pipeline Complete -- {elapsed/60:.1f} minutes")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print("=" * 65)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='start_from', type=int, default=0,
                        help='Resume from layer N (0=full run)')
    args = parser.parse_args()
    run_pipeline(start_from=args.start_from)
