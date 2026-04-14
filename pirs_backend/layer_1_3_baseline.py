"""
PIRS BACKEND - LAYERS 1-3: BASELINE RISK DETECTION
===================================================
Ensemble anomaly detection using:
- Isolation Forest (50%)
- LSTM Autoencoder (35%)
- One-Class SVM (15%)
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

# Import configuration
try:
    from config import PIRSConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from config import PIRSConfig

def load_processed_data():
    """
    Load preprocessed data and features.
    Prefers semantic features (40 curated) over raw PCA features (873).
    Set PIRSConfig.USE_SEMANTIC_FEATURES = False to use raw features.
    """
    print("\n[DIR] Loading processed data...")

    features_path  = os.path.join(PIRSConfig.OUTPUT_DIR, 'behavioral_features.npy')
    extracted_path = os.path.join(PIRSConfig.OUTPUT_DIR,
                                   PIRSConfig.EXTRACTED_FEATURES_FILE)
    semantic_path  = os.path.join(PIRSConfig.OUTPUT_DIR,
                                   PIRSConfig.SEMANTIC_FEATURES_FILE)
    raw_path       = os.path.join(PIRSConfig.OUTPUT_DIR, 'data_processed.csv')

    # Priority: raw-CERT extracted > semantic > original dayr6.2
    if os.path.exists(extracted_path):
        data_path = extracted_path
        print(f"   [OK] Using raw-CERT extracted features: "
              f"{PIRSConfig.EXTRACTED_FEATURES_FILE}")
    elif PIRSConfig.USE_SEMANTIC_FEATURES and os.path.exists(semantic_path):
        data_path = semantic_path
        print(f"   Using semantic features: {PIRSConfig.SEMANTIC_FEATURES_FILE}")
    elif os.path.exists(raw_path):
        data_path = raw_path
        print(f"   Using raw features: data_processed.csv")
        print(f"   [WARN]  Run data_extraction.py for best results")
    else:
        raise FileNotFoundError(
            "No data found. Run data_extraction.py first."
        )

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            "behavioral_features.npy not found. Run data_loading.py first."
        )

    df = pd.read_csv(data_path)
    behavioral_cols = np.load(features_path, allow_pickle=True).tolist()

    missing = [c for c in behavioral_cols if c not in df.columns]
    if missing:
        print(f"   [WARN]  {len(missing)} features not in file -- zero-filling")
        for c in missing:
            df[c] = 0.0

    print(f"[OK] Loaded {len(df):,} rows x {len(behavioral_cols)} features "
          f"({df['user'].nunique():,} users)")
    return df, behavioral_cols

def prepare_feature_matrix(df, behavioral_cols):
    """Prepare and scale feature matrix"""
    print("\n[SETUP] Preparing feature matrix...")
    
    X = df[behavioral_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"[OK] Feature matrix: {X.shape}")
    print(f"   NaN count: {np.isnan(X).sum()}")
    print(f"   Inf count: {np.isinf(X).sum()}")
    
    # Standardize
    print("\n[INFO] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[OK] Scaled: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")
    
    return X, X_scaled, scaler

def train_isolation_forest(X_scaled):
    """Train Isolation Forest model"""
    print("\n[TREE] Training Isolation Forest (50% weight)...")
    start = time.time()
    
    iso_forest = IsolationForest(
        n_estimators=PIRSConfig.ISOLATION_N_ESTIMATORS,
        contamination=PIRSConfig.ISOLATION_CONTAMINATION,
        random_state=PIRSConfig.RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    iso_forest.fit(X_scaled)
    iso_scores = iso_forest.decision_function(X_scaled)
    
    # Convert to 0-10 risk scale (invert: lower score = higher anomaly)
    iso_risk = 10 * (1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min()))
    
    elapsed = time.time() - start
    
    print(f"[OK] Isolation Forest: {elapsed:.1f}s")
    print(f"   Risk: [{iso_risk.min():.2f}, {iso_risk.max():.2f}], "
          f"u={iso_risk.mean():.2f}+/-{iso_risk.std():.2f}")
    
    return iso_forest, iso_risk

def train_lstm_autoencoder(X_scaled):
    """Train LSTM Autoencoder model"""
    print("\n[ML] Training LSTM Autoencoder (35% weight)...")
    start = time.time()
    
    n_features = X_scaled.shape[1]
    
    # Sample for training
    sample_size = min(PIRSConfig.LSTM_TRAIN_SAMPLE, len(X_scaled))
    sample_idx = np.random.RandomState(PIRSConfig.RANDOM_STATE).choice(
        len(X_scaled), sample_size, replace=False
    )
    X_train_sample = X_scaled[sample_idx]
    
    # Build model
    autoencoder = Sequential([
        Input(shape=(1, n_features)),
        LSTM(PIRSConfig.LSTM_LATENT_DIM, activation='relu'),
        RepeatVector(1),
        LSTM(PIRSConfig.LSTM_LATENT_DIM, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ], name='LSTM_Autoencoder')
    
    autoencoder.compile(optimizer='adam', loss='mae')
    
    print(f"   Training on {sample_size:,} samples...")
    
    # Train
    history = autoencoder.fit(
        X_train_sample.reshape(-1, 1, n_features),
        X_train_sample.reshape(-1, 1, n_features),
        epochs=PIRSConfig.LSTM_EPOCHS,
        batch_size=PIRSConfig.LSTM_BATCH_SIZE,
        verbose=0
    )
    
    print(f"   Final loss: {history.history['loss'][-1]:.4f}")
    
    # Predict in chunks
    print(f"   Computing reconstruction errors...")
    lstm_risk = []
    
    for i in range(0, len(X_scaled), PIRSConfig.CHUNK_SIZE):
        chunk = X_scaled[i:i+PIRSConfig.CHUNK_SIZE]
        chunk_reshaped = chunk.reshape(-1, 1, n_features)
        predictions = autoencoder.predict(chunk_reshaped, verbose=0)
        mse = np.mean((chunk_reshaped - predictions) ** 2, axis=(1, 2))
        lstm_risk.append(mse)
        
        if (i // PIRSConfig.CHUNK_SIZE + 1) % 10 == 0:
            print(f"   Processed {i+len(chunk):,} / {len(X_scaled):,}")
    
    lstm_risk = np.concatenate(lstm_risk)
    lstm_risk = 10 * (lstm_risk - lstm_risk.min()) / (lstm_risk.max() - lstm_risk.min())
    
    elapsed = time.time() - start
    
    print(f"[OK] LSTM Autoencoder: {elapsed:.1f}s")
    print(f"   Risk: [{lstm_risk.min():.2f}, {lstm_risk.max():.2f}], "
          f"u={lstm_risk.mean():.2f}+/-{lstm_risk.std():.2f}")
    
    return autoencoder, lstm_risk

def train_ocsvm(X_scaled):
    """Train One-Class SVM model"""
    print("\n[TARGET] Training One-Class SVM (15% weight)...")
    start = time.time()
    
    # Train on sample (SVM is expensive)
    sample_size = min(PIRSConfig.LSTM_TRAIN_SAMPLE, len(X_scaled))
    sample_idx = np.random.RandomState(PIRSConfig.RANDOM_STATE).choice(
        len(X_scaled), sample_size, replace=False
    )
    X_train_sample = X_scaled[sample_idx]
    
    ocsvm = OneClassSVM(
        kernel=PIRSConfig.OCSVM_KERNEL,
        gamma=PIRSConfig.OCSVM_GAMMA,
        nu=PIRSConfig.OCSVM_NU
    )
    
    ocsvm.fit(X_train_sample)
    svm_scores = ocsvm.decision_function(X_scaled)
    
    # Convert to 0-10 risk scale
    svm_risk = 10 * (1 - (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min()))
    
    elapsed = time.time() - start
    
    print(f"[OK] One-Class SVM: {elapsed:.1f}s")
    print(f"   Risk: [{svm_risk.min():.2f}, {svm_risk.max():.2f}], "
          f"u={svm_risk.mean():.2f}+/-{svm_risk.std():.2f}")
    
    return ocsvm, svm_risk

def compute_ensemble(iso_risk, lstm_risk, svm_risk):
    """Compute weighted ensemble risk score"""
    print("\n[MERGE] Computing ensemble risk scores...")
    
    ensemble_risk = (
        PIRSConfig.ISOLATION_FOREST_WEIGHT * iso_risk +
        PIRSConfig.LSTM_AUTOENCODER_WEIGHT * lstm_risk +
        PIRSConfig.OCSVM_WEIGHT * svm_risk
    )
    
    print(f"[OK] Ensemble Risk Statistics:")
    print(f"   Range: [{ensemble_risk.min():.2f}, {ensemble_risk.max():.2f}]")
    print(f"   Mean: {ensemble_risk.mean():.2f} +/- {ensemble_risk.std():.2f}")
    print(f"   Median: {np.median(ensemble_risk):.2f}")
    print(f"   95th percentile: {np.percentile(ensemble_risk, 95):.2f}")
    print(f"   99th percentile: {np.percentile(ensemble_risk, 99):.2f}")
    
    return ensemble_risk

def analyze_high_risk_users(df, ensemble_risk):
    """Analyze high-risk detection"""
    print(f"\n[ALERT] High-Risk Analysis (threshold >= {PIRSConfig.RISK_THRESHOLD_HIGH}):")
    
    high_risk_count = (ensemble_risk >= PIRSConfig.RISK_THRESHOLD_HIGH).sum()
    high_risk_pct = 100 * high_risk_count / len(ensemble_risk)
    
    print(f"   High-risk observations: {high_risk_count:,} ({high_risk_pct:.2f}%)")
    
    # User-level aggregation
    df_with_risk = df.copy()
    df_with_risk['risk_score'] = ensemble_risk
    
    user_max_risk = df_with_risk.groupby('user')['risk_score'].max().reset_index()
    high_risk_users = user_max_risk[user_max_risk['risk_score'] >= PIRSConfig.RISK_THRESHOLD_HIGH]
    
    print(f"   High-risk users: {len(high_risk_users):,} / {df['user'].nunique():,} "
          f"({100*len(high_risk_users)/df['user'].nunique():.1f}%)")
    
    # Show top 10
    print(f"\n   Top 10 users by risk:")
    top_users = user_max_risk.nlargest(10, 'risk_score')
    for i, row in top_users.iterrows():
        print(f"     User {row['user']}: {row['risk_score']:.2f}")
    
    return df_with_risk

def save_results(df, iso_risk, lstm_risk, svm_risk, ensemble_risk, 
                 iso_forest, autoencoder, ocsvm, scaler, behavioral_cols):
    """Save all outputs"""
    print("\n[SAVE] Saving results...")
    
    # Add risk scores to dataframe
    df['risk_iso'] = iso_risk
    df['risk_lstm'] = lstm_risk
    df['risk_svm'] = svm_risk
    df['risk_score'] = ensemble_risk
    
    # Save baseline results
    output_file = os.path.join(PIRSConfig.OUTPUT_DIR, PIRSConfig.OUTPUT_FILES['baseline'])
    # Use 'date' column if 'datetime' not present (data_extracted.csv uses 'date')
    date_col = 'datetime' if 'datetime' in df.columns else 'date'
    save_cols = ['user', 'day']
    if date_col in df.columns:
        save_cols.append(date_col)
    save_cols += ['risk_iso', 'risk_lstm', 'risk_svm', 'risk_score']
    if 'insider' in df.columns:
        save_cols.append('insider')
    df[save_cols].to_csv(
        output_file, index=False
    )
    print(f"[OK] Baseline results: {output_file}")
    
    # Save models
    models_file = os.path.join(PIRSConfig.OUTPUT_DIR, 'baseline_models.pkl')
    joblib.dump({
        'isolation_forest': iso_forest,
        'lstm_autoencoder': autoencoder,
        'ocsvm': ocsvm,
        'scaler': scaler,
        'behavioral_cols': behavioral_cols
    }, models_file)
    print(f"[OK] Models saved: {models_file}")
    
    return df

def run_baseline_detection():
    """Main function for baseline detection"""
    print("\n" + "="*70)
    print("LAYERS 1-3: BASELINE RISK DETECTION")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    df, behavioral_cols = load_processed_data()
    
    # Prepare features
    X, X_scaled, scaler = prepare_feature_matrix(df, behavioral_cols)
    
    # Train models
    iso_forest, iso_risk = train_isolation_forest(X_scaled)
    autoencoder, lstm_risk = train_lstm_autoencoder(X_scaled)
    ocsvm, svm_risk = train_ocsvm(X_scaled)
    
    # Ensemble
    ensemble_risk = compute_ensemble(iso_risk, lstm_risk, svm_risk)
    
    # Analyze
    df_with_risk = analyze_high_risk_users(df, ensemble_risk)
    
    # Save
    df_final = save_results(df_with_risk, iso_risk, lstm_risk, svm_risk, ensemble_risk,
                            iso_forest, autoencoder, ocsvm, scaler, behavioral_cols)
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print(f"[OK] BASELINE DETECTION COMPLETE")
    print(f"   Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("="*70 + "\n")
    
    return df_final

if __name__ == "__main__":
    try:
        df = run_baseline_detection()
        print("\n[OK] Baseline detection module executed successfully")
    except Exception as e:
        print(f"\n[ERROR] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)