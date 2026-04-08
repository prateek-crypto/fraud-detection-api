"""
train_models.py
---------------
Trains Logistic Regression, Random Forest, and XGBoost on the
preprocessed (SMOTE-balanced) training data.
Saves each model to models/<name>.pkl
"""
import os, joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42,
                                                      n_jobs=-1, class_weight="balanced"),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                             max_depth=6, scale_pos_weight=1,
                                             use_label_encoder=False,
                                             eval_metric="logloss", random_state=42,
                                             n_jobs=-1),
    }

def train_all(X_train, y_train, verbose=True) -> dict:
    """Train all models and return fitted model dict."""
    models = get_models()
    fitted = {}
    for name, model in models.items():
        if verbose:
            print(f"Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        fitted[name] = model
        path = os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".pkl")
        joblib.dump(model, path)
        if verbose:
            print(f"done  →  saved to {path}")
    return fitted

def load_all() -> dict:
    """Load all saved models from disk."""
    names = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest":       "random_forest.pkl",
        "XGBoost":             "xgboost.pkl",
    }
    fitted = {}
    for name, fname in names.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            fitted[name] = joblib.load(path)
        else:
            print(f"Warning: {path} not found — run train_all() first.")
    return fitted

# ── standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.preprocessing import load_data, preprocess
    from utils.evaluation import evaluate_model, metrics_summary_df

    DATA_PATH = "data/creditcard.csv"
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.\n"
              "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feat = preprocess(df)

    models = train_all(X_train, y_train)

    results = [evaluate_model(n, m, X_test, y_test) for n, m in models.items()]
    summary = metrics_summary_df(results)
    print("\n── Model comparison ──────────────────────────────────────────────")
    print(summary.to_string(index=False))
