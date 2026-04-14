"""
PIRS BACKEND - LAYER 4: PROSPECTIVE DRIFT DETECTION
====================================================
Detect behavioral escalation 7 days before threshold breach

"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

def load_baseline_results():
    """Load baseline risk scores"""
    print("\n[DIR] Loading baseline results...")
    
    baseline_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['baseline'])
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Baseline results not found. Run 02_layer_1_3_baseline.py first")
    
    df = pd.read_csv(baseline_file)
    # Handle both 'datetime' and 'date' column names
    if 'datetime' not in df.columns and 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        df['datetime'] = pd.to_datetime(df['day'], unit='D', origin='2010-01-01')
    df = df.sort_values(['user', 'day']).reset_index(drop=True)
    
    print(f"[OK] Loaded {len(df):,} observations for {df['user'].nunique():,} users")
    
    return df

def calculate_drift_for_user(user_data):
    """Calculate drift score for a single user"""
    user_data = user_data.sort_values('day')
    
    if len(user_data) < PIRSConfig.MIN_DAYS_FOR_DRIFT:
        return pd.DataFrame()  # Not enough data
    
    results = []
    
    for idx in range(len(user_data)):
        current_day = user_data.iloc[idx]
        current_day_num = current_day['day']
        
        # Get last DRIFT_WINDOW days
        window_start = current_day_num - PIRSConfig.DRIFT_WINDOW
        window_data = user_data[
            (user_data['day'] >= window_start) & 
            (user_data['day'] <= current_day_num)
        ]
        
        if len(window_data) < PIRSConfig.MIN_DAYS_FOR_DRIFT:
            continue
        
        # Linear regression on risk trajectory
        days = window_data['day'].values
        risks = window_data['risk_score'].values
        
        if len(days) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(days, risks)
            
            # Project forward
            future_day = current_day_num + PIRSConfig.FORECAST_HORIZON
            projected_risk = slope * future_day + intercept
            
            # Drift score (normalized slope)
            drift_score = max(0, min(1, slope * PIRSConfig.FORECAST_HORIZON / 10))
            
            # Early warning: will breach threshold?
            will_breach = (current_day['risk_score'] < PIRSConfig.RISK_THRESHOLD_HIGH and 
                          projected_risk >= PIRSConfig.RISK_THRESHOLD_HIGH)
            
            days_to_breach = None
            if slope > 0 and will_breach:
                # Calculate when it will breach
                days_to_breach = (PIRSConfig.RISK_THRESHOLD_HIGH - current_day['risk_score']) / slope
                days_to_breach = max(1, int(days_to_breach))
            
            results.append({
                'user': current_day['user'],
                'day': current_day_num,
                'datetime': current_day['datetime'],
                'risk_score': current_day['risk_score'],
                'drift_slope': slope,
                'drift_score': drift_score,
                'projected_risk_7d': projected_risk,
                'will_breach': will_breach,
                'days_to_breach': days_to_breach if will_breach else None,
                'r_squared': r_value**2
            })
    
    return pd.DataFrame(results)

def calculate_drift_all_users(df):
    """Calculate drift for all users"""
    print(f"\n[CHART] Calculating drift for {df['user'].nunique():,} users...")
    print(f"   Window: {PIRSConfig.DRIFT_WINDOW} days")
    print(f"   Forecast: {PIRSConfig.FORECAST_HORIZON} days ahead")
    
    start_time = time.time()
    
    all_drift = []
    users = df['user'].unique()
    
    for i, user in enumerate(users):
        user_data = df[df['user'] == user]
        user_drift = calculate_drift_for_user(user_data)
        
        if not user_drift.empty:
            all_drift.append(user_drift)
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(users) - i - 1) / rate if rate > 0 else 0
            print(f"   Processed {i+1:,} / {len(users):,} users ({rate:.0f} users/sec, {remaining:.0f}s remaining)")
    
    df_drift = pd.concat(all_drift, ignore_index=True)
    
    elapsed = time.time() - start_time
    print(f"\n[OK] Drift calculated: {len(df_drift):,} observations in {elapsed:.1f}s")
    
    return df_drift

def analyze_drift_results(df_drift):
    """Analyze drift detection results"""
    print(f"\n[SEARCH] Drift Detection Analysis:")
    
    # Overall stats
    print(f"   Mean drift score: {df_drift['drift_score'].mean():.3f}")
    print(f"   Median drift score: {df_drift['drift_score'].median():.3f}")
    print(f"   Max drift score: {df_drift['drift_score'].max():.3f}")
    
    # Drift categories
    low_drift = (df_drift['drift_score'] >= PIRSConfig.DRIFT_THRESHOLD_LOW) & \
                (df_drift['drift_score'] < PIRSConfig.DRIFT_THRESHOLD_MODERATE)
    mod_drift = (df_drift['drift_score'] >= PIRSConfig.DRIFT_THRESHOLD_MODERATE) & \
                (df_drift['drift_score'] < PIRSConfig.DRIFT_THRESHOLD_HIGH)
    high_drift = df_drift['drift_score'] >= PIRSConfig.DRIFT_THRESHOLD_HIGH
    
    print(f"\n   Drift categories:")
    print(f"     Low drift (>={PIRSConfig.DRIFT_THRESHOLD_LOW}): {low_drift.sum():,} ({100*low_drift.sum()/len(df_drift):.1f}%)")
    print(f"     Moderate drift (>={PIRSConfig.DRIFT_THRESHOLD_MODERATE}): {mod_drift.sum():,} ({100*mod_drift.sum()/len(df_drift):.1f}%)")
    print(f"     High drift (>={PIRSConfig.DRIFT_THRESHOLD_HIGH}): {high_drift.sum():,} ({100*high_drift.sum()/len(df_drift):.1f}%)")
    
    # Early warnings
    early_warnings = df_drift[df_drift['will_breach'] == True]
    print(f"\n   Early warnings: {len(early_warnings):,}")
    
    if len(early_warnings) > 0:
        avg_days_to_breach = early_warnings['days_to_breach'].mean()
        print(f"     Average days to breach: {avg_days_to_breach:.1f}")
        print(f"     Unique users warned: {early_warnings['user'].nunique():,}")
    
    # Top drift users
    print(f"\n   Top 10 users by drift:")
    user_max_drift = df_drift.groupby('user')['drift_score'].max().reset_index()
    top_drift = user_max_drift.nlargest(10, 'drift_score')
    for i, row in top_drift.iterrows():
        print(f"     User {row['user']}: drift={row['drift_score']:.3f}")

def save_drift_results(df_drift):
    """Save drift detection results"""
    print(f"\n[SAVE] Saving drift results...")
    
    output_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['drift'])
    df_drift.to_csv(output_file, index=False)
    
    print(f"[OK] Drift results saved: {output_file}")
    
    return output_file

def run_drift_detection():
    """Main function for drift detection"""
    print("\n" + "="*70)
    print("LAYER 4: PROSPECTIVE DRIFT DETECTION")
    print("="*70)
    
    start_time = time.time()
    
    # Load baseline results
    df = load_baseline_results()
    
    # Calculate drift
    df_drift = calculate_drift_all_users(df)
    
    # Analyze
    analyze_drift_results(df_drift)
    
    # Save
    save_drift_results(df_drift)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] DRIFT DETECTION COMPLETE")
    print(f"   Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("="*70 + "\n")
    
    return df_drift

if __name__ == "__main__":
    try:
        df = run_drift_detection()
        print("\n[OK] Drift detection module executed successfully")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)