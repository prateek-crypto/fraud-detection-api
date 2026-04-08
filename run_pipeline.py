"""
run_pipeline.py
---------------
One-click script that runs the full fraud detection pipeline:
  1. Load (or generate) data
  2. EDA summary
  3. Preprocess + SMOTE
  4. Train supervised models
  5. Evaluate all models
  6. Train anomaly detectors
  7. Train autoencoder
  8. Compute SHAP values
  9. Generate HTML report
  10. Print final summary table

Usage:
    python run_pipeline.py                     # uses data/creditcard.csv
    python run_pipeline.py --generate          # generates synthetic data first
    python run_pipeline.py --generate --rows 50000
    python run_pipeline.py --skip_autoencoder  # skip Keras (faster)
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def step(n, total, msg):
    print(f"\n{BOLD}{CYAN}[{n}/{total}] {msg}{RESET}")

def ok(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")

def warn(msg):
    print(f"  {YELLOW}⚠ {msg}{RESET}")

def banner(msg):
    w = 60
    print(f"\n{BOLD}{'─'*w}")
    print(f"  {msg}")
    print(f"{'─'*w}{RESET}")


TOTAL_STEPS = 9

def run(args):
    t0 = time.time()
    banner("Fraud Detection & Anomaly Detection — Full Pipeline")

    DATA_PATH = "data/creditcard.csv"

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    step(1, TOTAL_STEPS, "Data loading")
    if args.generate or not os.path.exists(DATA_PATH):
        if not args.generate:
            warn("creditcard.csv not found — generating synthetic data instead.")
        from data.generate_sample_data import generate_creditcard_like
        df = generate_creditcard_like(n_rows=args.rows, random_state=42)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        ok(f"Synthetic dataset generated: {len(df):,} rows")
    else:
        from utils.preprocessing import load_data
        df = load_data(DATA_PATH)
        ok(f"Dataset loaded: {len(df):,} rows")

    fraud_n  = df["Class"].sum()
    normal_n = (df["Class"] == 0).sum()
    ok(f"Class distribution — Normal: {normal_n:,} | Fraud: {fraud_n:,} ({fraud_n/len(df)*100:.3f}%)")

    # ── Step 2: EDA ──────────────────────────────────────────────────────────
    step(2, TOTAL_STEPS, "EDA summary")
    print(f"  Columns       : {list(df.columns[:5])} ... (total {df.shape[1]})")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Amount range  : €{df['Amount'].min():.2f} – €{df['Amount'].max():,.2f}")
    corr = df.corr()["Class"].abs().drop("Class")
    top5 = corr.nlargest(5)
    print(f"  Top 5 correlated features:")
    for feat, val in top5.items():
        print(f"    {feat:>5}  |r| = {val:.4f}")
    ok("EDA complete")

    # ── Step 3: Preprocess ───────────────────────────────────────────────────
    step(3, TOTAL_STEPS, "Preprocessing & SMOTE")
    from utils.preprocessing import preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    ok(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    ok(f"After SMOTE — fraud in train: {y_train.sum():,}/{len(y_train):,}")

    # ── Step 4: Train models ─────────────────────────────────────────────────
    step(4, TOTAL_STEPS, "Training supervised models")
    from models.train_models import train_all
    t_train = time.time()
    models = train_all(X_train, y_train, verbose=True)
    ok(f"All models trained in {time.time()-t_train:.1f}s")

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    step(5, TOTAL_STEPS, "Evaluating models")
    from utils.evaluation import evaluate_model, metrics_summary_df
    results = [evaluate_model(n, m, X_test, y_test) for n, m in models.items()]
    summary = metrics_summary_df(results)
    print()
    print(summary.to_string(index=False))
    best = summary.iloc[0]["Model"]
    ok(f"Best model: {best}  (AUC={summary.iloc[0]['ROC-AUC']})")

    # ── Step 6: Isolation Forest ─────────────────────────────────────────────
    step(6, TOTAL_STEPS, "Isolation Forest anomaly detection")
    from models.anomaly_detection import train_isolation_forest, isolation_forest_scores
    ifo    = train_isolation_forest(X_train.values)
    if_scores = isolation_forest_scores(ifo, X_test.values)
    fraud_score_mean  = if_scores[y_test == 1].mean()
    normal_score_mean = if_scores[y_test == 0].mean()
    ok(f"Fraud anomaly score  (mean): {fraud_score_mean:.4f}")
    ok(f"Normal anomaly score (mean): {normal_score_mean:.4f}")
    separation = (fraud_score_mean - normal_score_mean) / normal_score_mean * 100
    ok(f"Fraud score is {separation:.1f}% higher than normal — {'good' if separation > 20 else 'moderate'} separation")

    # ── Step 7: Autoencoder ──────────────────────────────────────────────────
    ae_model = ae_scaler = None
    if not args.skip_autoencoder:
        step(7, TOTAL_STEPS, "Autoencoder training")
        try:
            from models.anomaly_detection import train_autoencoder, autoencoder_scores
            y_tr_np  = y_train.values
            X_normal = X_train.values[y_tr_np == 0]
            split    = int(0.9 * len(X_normal))
            ae_model, ae_scaler, hist = train_autoencoder(
                X_normal[:split], X_normal[split:], epochs=args.ae_epochs
            )
            ae_scores = autoencoder_scores(ae_model, ae_scaler, X_test.values)
            ok(f"Autoencoder trained — val_loss: {min(hist.history['val_loss']):.6f}")
            ok(f"Fraud recon error (mean):  {ae_scores[y_test.values==1].mean():.6f}")
            ok(f"Normal recon error (mean): {ae_scores[y_test.values==0].mean():.6f}")
        except Exception as e:
            warn(f"Autoencoder skipped: {e}")
    else:
        step(7, TOTAL_STEPS, "Autoencoder — skipped (--skip_autoencoder)")
        warn("Pass --skip_autoencoder=False to enable")

    # ── Step 8: SHAP ─────────────────────────────────────────────────────────
    step(8, TOTAL_STEPS, "SHAP explainability")
    shap_values = shap_explainer = None
    try:
        import shap
        xgb_model   = models.get("XGBoost") or list(models.values())[-1]
        X_shap      = X_test.iloc[:args.shap_samples]
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_shap = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:5]
        ok("Top 5 features by mean |SHAP|:")
        for fname, val in top_shap:
            print(f"    {fname:>10}  {val:.4f}")
        shap_explainer = explainer
    except Exception as e:
        warn(f"SHAP skipped: {e}")

    # ── Step 9: HTML Report ───────────────────────────────────────────────────
    step(9, TOTAL_STEPS, "Generating HTML report")
    try:
        from utils.report_generator import generate_report
        report_path = generate_report(
            df=df,
            results=results,
            summary_df=summary,
            if_scores=if_scores,
            y_test=y_test,
            shap_values=shap_values,
            shap_X=X_test.iloc[:args.shap_samples] if shap_values is not None else None,
            feature_names=feature_names,
        )
        ok(f"Report saved to {report_path}")
    except Exception as e:
        warn(f"Report generation failed: {e}")

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  {GREEN}Dashboard : streamlit run app/streamlit_app.py{RESET}")
    print(f"  {GREEN}Report    : open reports/fraud_report.html{RESET}")
    print(f"  {GREEN}API       : python api/fraud_api.py{RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate",         action="store_true", help="Generate synthetic data")
    parser.add_argument("--rows",             type=int,   default=100_000)
    parser.add_argument("--skip_autoencoder", action="store_true")
    parser.add_argument("--ae_epochs",        type=int,   default=20)
    parser.add_argument("--shap_samples",     type=int,   default=300)
    args = parser.parse_args()
    run(args)
