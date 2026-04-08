"""
api/fraud_api.py
----------------
A lightweight Flask REST API that serves fraud predictions.

Endpoints:
  GET  /health               → service health check
  POST /predict              → predict fraud for a single transaction
  POST /predict/batch        → predict fraud for a batch (CSV or JSON)
  GET  /model/info           → loaded model metadata
  GET  /model/threshold      → current threshold
  POST /model/threshold      → update threshold
  GET  /stats                → request stats since startup

Run:
    python api/fraud_api.py
    python api/fraud_api.py --port 8080 --model xgboost
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, abort
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ── State ────────────────────────────────────────────────────────────────────
_model     = None
_threshold = 0.5
_model_name = ""
_stats     = {"requests": 0, "fraud_flagged": 0, "errors": 0, "start_time": time.time()}


def _load_model(model_name: str):
    global _model, _model_name
    paths = {
        "xgboost":              "models/xgboost.pkl",
        "random_forest":        "models/random_forest.pkl",
        "logistic_regression":  "models/logistic_regression.pkl",
    }
    key  = model_name.lower().replace(" ", "_")
    path = paths.get(key)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at '{path}'. "
            f"Run run_pipeline.py or train_models.py first."
        )
    _model      = joblib.load(path)
    _model_name = model_name
    print(f"[API] Loaded model: {model_name} from {path}")


def _preprocess_input(data: dict) -> pd.DataFrame:
    """
    Accept either:
      - raw transaction with Time, V1..V28, Amount (scales Amount/Time)
      - pre-scaled transaction with V1..V28, Amount_scaled, Time_scaled
    """
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

    # Scale Amount and Time if raw
    if "Amount" in df.columns and "Amount_scaled" not in df.columns:
        try:
            scaler = joblib.load("models/scaler.pkl")
            df["Amount_scaled"] = scaler.transform(df[["Amount"]])
            df.drop(columns=["Amount"], inplace=True)
        except Exception:
            df["Amount_scaled"] = (df["Amount"] - 88.35) / 250.12
            df.drop(columns=["Amount"], inplace=True)

    if "Time" in df.columns and "Time_scaled" not in df.columns:
        df["Time_scaled"] = (df["Time"] - 94813) / 47488
        df.drop(columns=["Time"], inplace=True)

    for col in ["Class", "label", "id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "healthy",
        "model":      _model_name,
        "model_loaded": _model is not None,
        "threshold":  _threshold,
        "uptime_sec": round(time.time() - _stats["start_time"], 1),
    })


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Predict fraud probability for a single transaction.

    Request body (JSON):
    {
      "Time": 12345,
      "V1": -1.36,
      ...
      "V28": 0.02,
      "Amount": 149.62
    }

    Response:
    {
      "fraud_probability": 0.023,
      "is_fraud": false,
      "confidence": "low",
      "threshold": 0.5
    }
    """
    if _model is None:
        return jsonify({"error": "No model loaded"}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        X = _preprocess_input(data)
        prob = float(_model.predict_proba(X)[0][1])
        is_fraud = prob >= _threshold

        _stats["requests"] += 1
        if is_fraud:
            _stats["fraud_flagged"] += 1

        return jsonify({
            "fraud_probability": round(prob, 6),
            "is_fraud":          is_fraud,
            "confidence":        (
                "very high" if prob > 0.9 else
                "high"      if prob > 0.7 else
                "medium"    if prob > 0.4 else
                "low"
            ),
            "threshold": _threshold,
            "model":     _model_name,
        })

    except Exception as e:
        _stats["errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict fraud for multiple transactions.

    Accepts:
      - JSON array: [{"V1": ..., "Amount": ...}, ...]
      - CSV file upload: multipart/form-data with field 'file'

    Returns predictions for each row.
    """
    if _model is None:
        return jsonify({"error": "No model loaded"}), 503

    try:
        content_type = request.content_type or ""

        if "multipart" in content_type:
            if "file" not in request.files:
                return jsonify({"error": "No file in request"}), 400
            f = request.files["file"]
            df_raw = pd.read_csv(f)
        else:
            data = request.get_json(force=True)
            df_raw = pd.DataFrame(data if isinstance(data, list) else [data])

        has_labels = "Class" in df_raw.columns
        y_true     = df_raw["Class"].values if has_labels else None
        X          = _preprocess_input(df_raw.to_dict(orient="records"))
        probs      = _model.predict_proba(X)[:, 1]
        preds      = (probs >= _threshold).astype(int)

        _stats["requests"] += len(X)
        _stats["fraud_flagged"] += int(preds.sum())

        response = {
            "n_transactions":  len(X),
            "n_fraud_flagged": int(preds.sum()),
            "fraud_rate":      round(float(preds.mean()), 4),
            "threshold":       _threshold,
            "predictions": [
                {
                    "index":             i,
                    "fraud_probability": round(float(p), 6),
                    "is_fraud":          bool(preds[i]),
                }
                for i, p in enumerate(probs)
            ],
        }

        if has_labels:
            from sklearn.metrics import roc_auc_score, f1_score
            response["evaluation"] = {
                "auc": round(float(roc_auc_score(y_true, probs)), 4),
                "f1":  round(float(f1_score(y_true, preds, zero_division=0)), 4),
                "true_fraud_in_batch": int(y_true.sum()),
                "correctly_flagged":   int((preds == 1) & (y_true == 1)).sum() if hasattr(y_true, '__iter__') else None,
            }

        return jsonify(response)

    except Exception as e:
        _stats["errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 503
    info = {
        "model_name":  _model_name,
        "model_type":  type(_model).__name__,
        "threshold":   _threshold,
    }
    try:
        info["n_features"] = _model.n_features_in_
    except Exception:
        pass
    return jsonify(info)


@app.route("/model/threshold", methods=["GET", "POST"])
def threshold():
    global _threshold
    if request.method == "GET":
        return jsonify({"threshold": _threshold})
    data = request.get_json(force=True)
    t    = float(data.get("threshold", 0.5))
    if not 0 < t < 1:
        return jsonify({"error": "Threshold must be between 0 and 1"}), 400
    _threshold = t
    return jsonify({"threshold": _threshold, "message": f"Threshold updated to {_threshold}"})


@app.route("/stats", methods=["GET"])
def stats():
    uptime = time.time() - _stats["start_time"]
    return jsonify({
        "total_requests":  _stats["requests"],
        "fraud_flagged":   _stats["fraud_flagged"],
        "errors":          _stats["errors"],
        "fraud_flag_rate": round(_stats["fraud_flagged"] / max(1, _stats["requests"]), 4),
        "uptime_seconds":  round(uptime, 1),
        "requests_per_min": round(_stats["requests"] / max(1, uptime) * 60, 2),
    })


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int,  default=5000)
    parser.add_argument("--host",  type=str,  default="0.0.0.0")
    parser.add_argument("--model", type=str,  default="xgboost",
                        choices=["xgboost", "random_forest", "logistic_regression"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    _threshold = args.threshold

    try:
        _load_model(args.model)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] API will start without a model. Train models first.")

    print(f"\n[API] Starting Fraud Detection API")
    print(f"[API] Host      : http://{args.host}:{args.port}")
    print(f"[API] Model     : {_model_name or 'none'}")
    print(f"[API] Threshold : {_threshold}")
    print(f"[API] Endpoints :")
    print(f"       GET  /health")
    print(f"       POST /predict")
    print(f"       POST /predict/batch")
    print(f"       GET  /model/info")
    print(f"       GET  /stats\n")

    app.run(host=args.host, port=args.port, debug=False)
