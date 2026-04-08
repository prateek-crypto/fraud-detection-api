"""
anomaly_detection.py
--------------------
Unsupervised anomaly detection:
  - Isolation Forest
  - DBSCAN (on t-SNE reduced data for speed)
  - Autoencoder (Keras)
"""
import os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Isolation Forest ──────────────────────────────────────────────────────────

def train_isolation_forest(X_train, contamination=0.002, n_estimators=200):
    """
    contamination ≈ fraction of fraud in original dataset.
    Lower = stricter (fewer flagged as anomalies).
    """
    ifo = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    ifo.fit(X_train)
    joblib.dump(ifo, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    print("Isolation Forest trained and saved.")
    return ifo

def isolation_forest_scores(model, X):
    """
    Returns anomaly scores in [0,1] where higher = more anomalous.
    decision_function returns negative → flip and normalise.
    """
    raw = model.decision_function(X)
    scores = 1 - (raw - raw.min()) / (raw.max() - raw.min())
    return scores


# ── DBSCAN ────────────────────────────────────────────────────────────────────

def run_dbscan(X, eps=0.5, min_samples=10, n_tsne_components=2,
               sample_size=10000, random_state=42):
    """
    Runs DBSCAN on a t-SNE projection of the data.
    Returns labels and the 2-D projection (for plotting).
    Points labelled -1 are noise/anomalies.
    Uses a sample for speed.
    """
    idx = np.random.default_rng(random_state).choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values

    print("Running t-SNE (this may take ~1-2 min)...")
    tsne = TSNE(n_components=n_tsne_components, random_state=random_state,
                perplexity=30, n_iter=500, verbose=0)
    X_2d = tsne.fit_transform(X_sample)

    print("Running DBSCAN...")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_2d)

    n_noise  = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN: {n_clusters} cluster(s), {n_noise} noise points")

    return labels, X_2d, idx


# ── Autoencoder ───────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int):
    """
    Lightweight autoencoder that learns the structure of NORMAL transactions.
    Fraud → high reconstruction error.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required for the autoencoder: pip install tensorflow")

    tf.random.set_seed(42)

    inputs  = keras.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(32, activation="relu")(inputs)
    encoded = keras.layers.Dense(16, activation="relu")(encoded)
    encoded = keras.layers.Dense(8,  activation="relu")(encoded)

    decoded = keras.layers.Dense(16, activation="relu")(encoded)
    decoded = keras.layers.Dense(32, activation="relu")(decoded)
    outputs = keras.layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.Model(inputs, outputs, name="fraud_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(X_train_normal, X_val_normal, epochs=30, batch_size=256):
    """
    Train on NORMAL transactions only.
    X_train_normal, X_val_normal: numpy arrays of normal rows.
    """
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_train_normal)
    X_va = scaler.transform(X_val_normal)

    model = build_autoencoder(X_tr.shape[1])

    try:
        import tensorflow as tf
        cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                                monitor="val_loss")]
        history = model.fit(
            X_tr, X_tr,
            validation_data=(X_va, X_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=1,
        )
    except Exception as e:
        print(f"Autoencoder training error: {e}")
        raise

    model.save(os.path.join(MODEL_DIR, "autoencoder.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "ae_scaler.pkl"))
    print("Autoencoder saved.")
    return model, scaler, history


def autoencoder_scores(model, scaler, X):
    """Reconstruction MSE per sample — higher = more anomalous."""
    X_scaled = scaler.transform(X)
    X_recon  = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - X_recon, 2), axis=1)
    return mse


def load_autoencoder():
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.keras"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "ae_scaler.pkl"))
        return model, scaler
    except Exception as e:
        print(f"Could not load autoencoder: {e}")
        return None, None


# ── standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.preprocessing import load_data, preprocess
    from utils.evaluation import evaluate_model, plot_anomaly_scores, plot_tsne

    DATA_PATH = "data/creditcard.csv"
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feat = preprocess(df)

    # Isolation Forest
    ifo    = train_isolation_forest(X_train)
    scores = isolation_forest_scores(ifo, X_test)
    print("IF anomaly scores — fraud mean:", scores[y_test == 1].mean().round(4))
    print("IF anomaly scores — normal mean:", scores[y_test == 0].mean().round(4))

    # Autoencoder
    X_tr_np  = X_train.values if hasattr(X_train, "values") else X_train
    X_te_np  = X_test.values  if hasattr(X_test,  "values") else X_test
    y_tr_np  = y_train.values if hasattr(y_train, "values") else y_train

    X_tr_normal = X_tr_np[y_tr_np == 0]
    split = int(0.9 * len(X_tr_normal))
    ae_model, ae_scaler, _ = train_autoencoder(X_tr_normal[:split], X_tr_normal[split:])

    ae_scores = autoencoder_scores(ae_model, ae_scaler, X_te_np)
    print("AE recon error — fraud mean:",  ae_scores[y_test.values == 1].mean().round(4))
    print("AE recon error — normal mean:", ae_scores[y_test.values == 0].mean().round(4))
