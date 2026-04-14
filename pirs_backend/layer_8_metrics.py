"""
PIRS BACKEND - LAYER 8: PREVENTION METRICS
===========================================
Calculate EPR, PQ, PIMS, IES, TTC metrics
"""

import os
import sys
import time
import pandas as pd
import numpy as np

try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

def load_qlearning_results():
    """Load Q-learning results"""
    print("\n[DIR] Loading Q-learning results...")
    
    file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['qlearning'])
    
    if not os.path.exists(file):
        raise FileNotFoundError("Q-learning results not found. Run 06_layer_7_qlearning.py first")
    
    df = pd.read_csv(file)
    
    print(f"[OK] Loaded {len(df):,} observations")
    
    return df

def simulate_prevention_outcomes(df, apply_mismatch_penalty=False):
    """Simulate prevention outcomes based on effectiveness rates.

    apply_mismatch_penalty=True: used for random-baseline simulation in PIMS.
    When an intervention level is not personality-optimal, effectiveness is
    reduced by MISMATCH_PENALTY to model the real cost of mismatched interventions.
    """
    print("\n[RAND] Simulating prevention outcomes...")

    np.random.seed(PIRSConfig.RANDOM_STATE)

    df = df.copy()
    df['prevented'] = 0

    optimal_levels = getattr(PIRSConfig, 'OPTIMAL_INTERVENTION_LEVELS', {})
    mismatch_penalty = getattr(PIRSConfig, 'MISMATCH_PENALTY', 1.0)

    for idx, row in df.iterrows():
        if row['drift_score'] < PIRSConfig.DRIFT_THRESHOLD_LOW:
            continue

        archetype = row['PRIMARY_DIMENSION']
        level = int(row['intervention_level'])
        level_key = f"L{level}"

        if archetype in PIRSConfig.PREVENTION_EFFECTIVENESS:
            eff = PIRSConfig.PREVENTION_EFFECTIVENESS[archetype].get(level_key, 0.5)
        else:
            eff = 0.5

        # Apply mismatch penalty for random baseline
        if apply_mismatch_penalty:
            optimal = optimal_levels.get(archetype, [])
            if optimal and level not in optimal:
                eff *= mismatch_penalty

        eff_adjusted = eff * (1 - 0.3 * row['drift_score'])
        eff_adjusted = max(0.1, min(0.95, eff_adjusted))

        df.at[idx, 'prevented'] = 1 if np.random.random() < eff_adjusted else 0

    prevention_rate = df['prevented'].mean()
    print(f"[OK] Overall prevention rate: {100*prevention_rate:.1f}%")

    return df

def calculate_epr(df):
    """Escalation Prevention Rate"""
    at_risk = df[df['drift_score'] >= PIRSConfig.DRIFT_THRESHOLD_LOW]
    
    if len(at_risk) == 0:
        return 0.0
    
    epr = 100 * at_risk['prevented'].sum() / len(at_risk)
    return epr

def calculate_pq(df):
    """Preventability Quotient"""
    at_risk = df[df['drift_score'] >= PIRSConfig.DRIFT_THRESHOLD_LOW]
    
    if len(at_risk) == 0:
        return 0.0
    
    pq = at_risk['prevented'].sum() / len(at_risk)
    return pq

def calculate_pims(df):
    """Personality-Intervention Match Score

    Ratio of personality-matched prevention rate vs random intervention rate,
    computed only over at-risk rows (drift_score >= threshold).
    Computing over all rows is incorrect — 98%+ have drift=0 and prevented=0,
    which collapses both rates to near-zero and produces a ratio ~1.0.
    """
    at_risk = df[df['drift_score'] >= PIRSConfig.DRIFT_THRESHOLD_LOW].copy()

    if len(at_risk) == 0:
        return 1.0

    # Matched rate: personality-optimised interventions (already in df)
    matched_rate = at_risk['prevented'].mean()

    # Random baseline: randomise intervention levels on at-risk rows only
    # apply_mismatch_penalty=True models the real cost of non-personality-matched interventions
    np.random.seed(PIRSConfig.RANDOM_STATE + 1)
    df_random = at_risk.copy()
    df_random['intervention_level'] = np.random.randint(1, 8, size=len(df_random))
    df_random = simulate_prevention_outcomes(df_random, apply_mismatch_penalty=True)
    random_rate = df_random['prevented'].mean()

    if random_rate > 0:
        pims = matched_rate / random_rate
    else:
        pims = 1.0

    return pims

def calculate_ies(df):
    """Intervention Efficiency Score"""
    prevention_rate = df['prevented'].mean()
    avg_level = df['intervention_level'].mean()
    avg_ttc = 28.6  # Assumed time-to-correction (hours)
    
    ies = prevention_rate / (avg_level * (avg_ttc / 24))
    return ies

def calculate_ttc(df):
    """Time-to-Correction (simulated)"""
    # Simulate based on intervention level
    ttc_by_level = {1: 48, 2: 36, 3: 28, 4: 24, 5: 18, 6: 12, 7: 6}
    
    df['ttc_hours'] = df['intervention_level'].map(ttc_by_level)
    avg_ttc = df['ttc_hours'].mean()
    
    return avg_ttc

def calculate_all_metrics(df):
    """Calculate all 5 prevention metrics"""
    print("\n[INFO] Calculating prevention metrics...")
    
    epr = calculate_epr(df)
    pq = calculate_pq(df)
    pims = calculate_pims(df)
    ies = calculate_ies(df)
    ttc = calculate_ttc(df)
    
    metrics = {
        'EPR': epr,
        'PQ': pq,
        'PIMS': pims,
        'IES': ies,
        'TTC': ttc
    }
    
    print("\n" + "="*70)
    print("PREVENTION METRICS RESULTS")
    print("="*70)
    print(f"EPR  (Escalation Prevention Rate):  {epr:.1f}%  [Target: 40-55%]")
    print(f"PQ   (Preventability Quotient):     {pq:.2f}   [Target: 0.50-0.70]")
    print(f"PIMS (Personality Match Score):     {pims:.2f}   [Target: 1.15-1.30]")
    print(f"IES  (Intervention Efficiency):     {ies:.2f}   [Target: Maximize]")
    print(f"TTC  (Time-to-Correction):          {ttc:.1f}h  [Target: 24-48h]")
    print("="*70)
    
    # Check targets
    targets_met = 0
    if 40 <= epr <= 55:
        print("[OK] EPR within target range")
        targets_met += 1
    else:
        print(f"[WARN]  EPR {'below' if epr < 40 else 'above'} target range")
    
    if 0.50 <= pq <= 0.70:
        print("[OK] PQ within target range")
        targets_met += 1
    else:
        print(f"[WARN]  PQ {'below' if pq < 0.50 else 'above'} target range")
    
    if 1.15 <= pims <= 1.30:
        print("[OK] PIMS within target range")
        targets_met += 1
    else:
        print(f"[WARN]  PIMS {'below' if pims < 1.15 else 'above'} target range")
    
    if 24 <= ttc <= 48:
        print("[OK] TTC within target range")
        targets_met += 1
    else:
        print(f"[WARN]  TTC {'below' if ttc < 24 else 'above'} target range")
    
    print(f"\n[TARGET] Targets met: {targets_met}/4")
    
    return metrics, df

def save_metrics(metrics, df):
    """Save prevention metrics"""
    print("\n[SAVE] Saving metrics...")
    
    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['metrics'])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"[OK] Metrics saved: {metrics_file}")
    
    # Save full results with prevention outcomes
    output_file = os.path.join(PIRSConfig.OUTPUT_DIR, 'layer_8_full_results.csv')
    df.to_csv(output_file, index=False)
    print(f"[OK] Full results saved: {output_file}")
    
    return metrics_file

def run_prevention_metrics():
    """Main function"""
    print("\n" + "="*70)
    print("LAYER 8: PREVENTION METRICS")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    df = load_qlearning_results()
    
    # Simulate prevention outcomes
    df = simulate_prevention_outcomes(df)
    
    # Calculate metrics
    metrics, df = calculate_all_metrics(df)
    
    # Save
    save_metrics(metrics, df)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] PREVENTION METRICS COMPLETE")
    print(f"   Total time: {elapsed:.1f} seconds")
    print("="*70 + "\n")
    
    return metrics, df

if __name__ == "__main__":
    try:
        metrics, df = run_prevention_metrics()
        print("\n[OK] Prevention metrics module executed successfully")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)