"""
PIRS V2 - LAYER 4: ENSEMBLE ANOMALY SCORING
=============================================
Runs three anomaly detectors on the DEVIATION features (not raw features).

Using deviation features instead of raw features means the models learn
"what does abnormal-for-you look like" rather than "what does unusual
behaviour look like globally."

Models:
  1. Isolation Forest (50%) -- tree-based, fast, good for high-dimensional
  2. LSTM Autoencoder  (35%) -- temporal, captures sequential patterns
  3. One-Class SVM     (15%) -- kernel-based, catches non-linear boundaries

Output: anomaly_score per user-day (0-10 scale)
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ModelConfig

cfg = ModelConfig()


# ---------------------------------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------------------------------

def run_isolation_forest(X: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest
    print("  [IsoForest] Training...")
    model = IsolationForest(
        n_estimators=cfg.ISOFOREST_N_ESTIMATORS,
        contamination=cfg.ISOFOREST_CONTAMINATION,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X)
    # decision_function: negative = more anomalous
    raw = -model.decision_function(X)
    # Normalize to 0-10
    r_min, r_max = raw.min(), raw.max()
    if r_max > r_min:
        scores = 10 * (raw - r_min) / (r_max - r_min)
    else:
        scores = np.zeros(len(raw))
    print(f"  [IsoForest] Done. mean={scores.mean():.3f}")
    return scores


# ---------------------------------------------------------------------------
# LSTM AUTOENCODER
# ---------------------------------------------------------------------------

def run_lstm_autoencoder(df: pd.DataFrame, dev_cols: list) -> np.ndarray:
    """
    Trains an LSTM autoencoder on sequences of deviation vectors.
    Reconstruction error = anomaly score.
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow import keras
    except ImportError:
        print("  [LSTM] TensorFlow not available, skipping LSTM")
        return np.zeros(len(df))

    print("  [LSTM] Building sequences...")

    SEQ_LEN = 7    # 7-day sequences

    # Build per-user sequences sorted by day
    sequences = []
    row_indices = []

    for user, udf in df.groupby('user'):
        udf = udf.sort_values('day').reset_index()
        vals = udf[dev_cols].values.astype(float)
        idxs = udf['index'].values

        for i in range(len(vals)):
            start = max(0, i - SEQ_LEN + 1)
            seq   = vals[start: i + 1]
            # Pad if shorter than SEQ_LEN
            if len(seq) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])
            sequences.append(seq)
            row_indices.append(idxs[i])

    X_seq = np.array(sequences, dtype=np.float32)
    n_features = X_seq.shape[2]

    # Sample for training
    n_train = min(cfg.LSTM_TRAIN_SAMPLE, len(X_seq))
    train_idx = np.random.RandomState(cfg.RANDOM_STATE).choice(
        len(X_seq), n_train, replace=False
    )
    X_train = X_seq[train_idx]

    print(f"  [LSTM] Training on {n_train:,} sequences "
          f"({SEQ_LEN} steps x {n_features} features)...")

    # Build autoencoder
    inputs  = keras.Input(shape=(SEQ_LEN, n_features))
    encoded = keras.layers.LSTM(cfg.LSTM_LATENT_DIM, return_sequences=False)(inputs)
    repeated = keras.layers.RepeatVector(SEQ_LEN)(encoded)
    decoded = keras.layers.LSTM(cfg.LSTM_LATENT_DIM, return_sequences=True)(repeated)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(decoded)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train,
              epochs=cfg.LSTM_EPOCHS,
              batch_size=cfg.LSTM_BATCH_SIZE,
              verbose=0)

    # Reconstruction error for all sequences
    print("  [LSTM] Computing reconstruction errors...")
    X_pred = model.predict(X_seq, batch_size=512, verbose=0)
    recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

    # Map back to original row order
    scores_dict = dict(zip(row_indices, recon_error))
    all_errors = np.array([scores_dict.get(i, 0.0) for i in range(len(df))])

    # Normalize to 0-10
    r_min, r_max = all_errors.min(), all_errors.max()
    if r_max > r_min:
        scores = 10 * (all_errors - r_min) / (r_max - r_min)
    else:
        scores = np.zeros(len(all_errors))

    print(f"  [LSTM] Done. mean={scores.mean():.3f}")
    return scores


# ---------------------------------------------------------------------------
# ONE-CLASS SVM
# ---------------------------------------------------------------------------

def run_ocsvm(X: np.ndarray) -> np.ndarray:
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler

    print("  [OC-SVM] Training (subsampling for speed)...")

    # OC-SVM is slow on large data -- subsample for training
    n_train = min(20_000, len(X))
    train_idx = np.random.RandomState(cfg.RANDOM_STATE).choice(
        len(X), n_train, replace=False
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(
        nu=cfg.OCSVM_NU,
        kernel=cfg.OCSVM_KERNEL,
        gamma=cfg.OCSVM_GAMMA
    )
    model.fit(X_scaled[train_idx])

    raw = -model.decision_function(X_scaled)
    r_min, r_max = raw.min(), raw.max()
    if r_max > r_min:
        scores = 10 * (raw - r_min) / (r_max - r_min)
    else:
        scores = np.zeros(len(raw))

    print(f"  [OC-SVM] Done. mean={scores.mean():.3f}")
    return scores


# ---------------------------------------------------------------------------
# ENSEMBLE
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, dev_cols: list) -> pd.DataFrame:
    """
    Run ensemble anomaly detection on deviation features.

    Args:
        df:       dataframe with deviation columns
        dev_cols: list of deviation column names

    Returns:
        df with added columns: score_iso, score_lstm, score_svm, anomaly_score
    """
    print(f"\n[L4] Ensemble anomaly scoring on {len(df):,} rows, "
          f"{len(dev_cols)} deviation features...")

    X = df[dev_cols].fillna(0).values.astype(float)

    df['score_iso']  = run_isolation_forest(X)
    df['score_lstm'] = run_lstm_autoencoder(df, dev_cols)
    df['score_svm']  = run_ocsvm(X)

    # Weighted ensemble
    df['anomaly_score'] = (
        cfg.ISOFOREST_WEIGHT * df['score_iso'] +
        cfg.LSTM_WEIGHT      * df['score_lstm'] +
        cfg.OCSVM_WEIGHT     * df['score_svm']
    )

    print(f"\n  Ensemble score: mean={df['anomaly_score'].mean():.3f}, "
          f"max={df['anomaly_score'].max():.3f}")
    print(f"  Users with score > 6.0: "
          f"{(df.groupby('user')['anomaly_score'].max() >= 6.0).sum():,}")

    return df
