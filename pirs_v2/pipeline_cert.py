"""
PIRS V2 - CERT PIPELINE RUNNER
================================
Runs the full 9-layer PIRS framework on CERT r6.2 dataset.

Usage:
    python pipeline_cert.py              # Full run
    python pipeline_cert.py --from 3    # Resume from layer 3

Layers:
    1. Personal behavioral baseline
    2. Daily deviation from baseline
    3. Behavioral drift detection
    4. Ensemble anomaly scoring
    5. 7/14-day breach prediction
    6. Personality context
    7. Personality-matched intervention
    8. Q-learning optimization
    9. Metrics + SHAP explainability
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import CERTConfig, ModelConfig

cfg  = CERTConfig()
mcfg = ModelConfig()

# Layer imports
from extractors.cert_extractor  import run_cert_extraction
from core.layer_1_baseline      import run as run_baseline, get_feature_cols
from core.layer_2_deviation     import run as run_deviation
from core.layer_3_drift         import run as run_drift
from core.layer_4_anomaly       import run as run_anomaly
from core.layer_5_prediction    import run as run_prediction, evaluate_early_warning
from core.layer_6_personality   import run as run_personality
from core.layer_7_intervention  import run as run_intervention
from core.layer_8_rl            import run as run_rl
from core.layer_9_metrics       import run as run_metrics

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILES = {
    0: os.path.join(cfg.OUTPUT_DIR, 'cert_features.csv'),
    3: os.path.join(cfg.OUTPUT_DIR, 'cert_after_drift.csv'),
    4: os.path.join(cfg.OUTPUT_DIR, 'cert_after_anomaly.csv'),
    5: os.path.join(cfg.OUTPUT_DIR, 'cert_after_prediction.csv'),
    7: os.path.join(cfg.OUTPUT_DIR, 'cert_after_intervention.csv'),
    9: os.path.join(cfg.OUTPUT_DIR, 'cert_complete.csv'),
}

PERSONALITY_FILE  = os.path.join(cfg.OUTPUT_DIR, 'cert_personality.csv')
EARLY_WARNING_FILE = os.path.join(cfg.OUTPUT_DIR, 'cert_early_warning.csv')
METRICS_FILE      = os.path.join(cfg.OUTPUT_DIR, 'cert_metrics.csv')
SHAP_FILE         = os.path.join(cfg.OUTPUT_DIR, 'cert_shap.csv')


def checkpoint_exists(layer: int) -> bool:
    return os.path.exists(CHECKPOINT_FILES.get(layer, ''))


def save(df: pd.DataFrame, layer: int):
    path = CHECKPOINT_FILES.get(layer)
    if path:
        df.to_csv(path, index=False)
        print(f"  [SAVED] {os.path.basename(path)} ({len(df):,} rows)")


def load(layer: int) -> pd.DataFrame:
    path = CHECKPOINT_FILES[layer]
    print(f"  [LOAD] {os.path.basename(path)}")
    return pd.read_csv(path, low_memory=False)


def run_pipeline(start_from: int = 0):
    print("=" * 65)
    print("  PIRS V2 -- CERT r6.2 Pipeline")
    print("  Pre-incident insider threat detection (9 layers)")
    print("=" * 65)

    t_start = time.time()
    dev_cols = None

    # ----------------------------------------------------------------
    # LAYER 0: EXTRACTION
    # ----------------------------------------------------------------
    if start_from <= 0:
        if checkpoint_exists(0):
            print("\n[L0] Extraction checkpoint found -- loading...")
            df = load(0)
        else:
            df = run_cert_extraction()
    else:
        # Load from closest available checkpoint
        for ck in sorted(CHECKPOINT_FILES.keys(), reverse=True):
            if ck <= start_from and checkpoint_exists(ck):
                df = load(ck)
                print(f"  Resuming from layer {ck} checkpoint")
                break

    feature_cols = get_feature_cols(df)
    print(f"\n  Feature columns: {len(feature_cols)}")

    # ----------------------------------------------------------------
    # LAYERS 1-2: BASELINE + DEVIATION
    # ----------------------------------------------------------------
    if start_from <= 2:
        print("\n" + "-"*40)
        baseline, feature_cols = run_baseline(df, feature_cols)
        df, dev_cols = run_deviation(df, baseline, feature_cols)
    else:
        # dev_cols need to be reconstructed
        dev_cols = [c for c in df.columns if c.endswith('_dev')]
        print(f"  Loaded {len(dev_cols)} deviation columns")

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
    # LAYER 4: ANOMALY SCORING
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

        # Early warning evaluation against 5 real insiders
        ew = evaluate_early_warning(df, cfg.INSIDER_MALICIOUS_DAYS)
        ew.to_csv(EARLY_WARNING_FILE, index=False)
        print(f"  [SAVED] {os.path.basename(EARLY_WARNING_FILE)}")
    elif start_from > 5:
        df = load(5)
        dev_cols = [c for c in df.columns if c.endswith('_dev')]

    # ----------------------------------------------------------------
    # LAYER 6: PERSONALITY
    # ----------------------------------------------------------------
    if start_from <= 6:
        print("\n" + "-"*40)
        personality_df, df = run_personality(df)
        personality_df.to_csv(PERSONALITY_FILE, index=False)
        print(f"  [SAVED] {os.path.basename(PERSONALITY_FILE)}")

    # ----------------------------------------------------------------
    # LAYER 7: INTERVENTION
    # ----------------------------------------------------------------
    if start_from <= 7:
        print("\n" + "-"*40)
        df = run_intervention(df)
        save(df, 7)

    # ----------------------------------------------------------------
    # LAYER 8: Q-LEARNING
    # ----------------------------------------------------------------
    if start_from <= 8:
        print("\n" + "-"*40)
        df, agent = run_rl(df)

    # ----------------------------------------------------------------
    # LAYER 9: METRICS + SHAP
    # ----------------------------------------------------------------
    if start_from <= 9:
        print("\n" + "-"*40)
        if dev_cols is None:
            dev_cols = [c for c in df.columns if c.endswith('_dev')]

        metrics = run_metrics(
            df,
            dev_cols,
            malicious_days=cfg.INSIDER_MALICIOUS_DAYS,
            n_users=cfg.N_USERS,
        )

        # Save all outputs
        save(df, 9)

        metrics['summary_df'].to_csv(METRICS_FILE, index=False)
        print(f"  [SAVED] {os.path.basename(METRICS_FILE)}")

        if not metrics.get('shap', pd.DataFrame()).empty:
            metrics['shap'].to_csv(SHAP_FILE, index=False)
            print(f"  [SAVED] {os.path.basename(SHAP_FILE)}")

    # ----------------------------------------------------------------
    # DONE
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  CERT Pipeline Complete -- {elapsed/60:.1f} minutes")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print("=" * 65)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='start_from', type=int, default=0,
                        help='Resume from layer N (0=full run)')
    args = parser.parse_args()
    run_pipeline(start_from=args.start_from)
