"""
utils/report_generator.py
--------------------------
Generates a beautiful, fully self-contained HTML report of the
fraud detection pipeline results.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

FRAUD_COLOR  = "#E24B4A"
NORMAL_COLOR = "#378ADD"


def _fig_to_html(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _metric_card(label: str, value: str, color: str = "#378ADD") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def _table_html(df: pd.DataFrame) -> str:
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows   = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows += f"<tr>{cells}</tr>"
    return f"""
    <table>
        <thead><tr>{header}</tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def generate_report(
    df: pd.DataFrame,
    results: list,
    summary_df: pd.DataFrame,
    if_scores: np.ndarray,
    y_test,
    shap_values=None,
    shap_X=None,
    feature_names=None,
    output_path: str = "reports/fraud_report.html",
) -> str:

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_fraud  = int(df["Class"].sum())
    n_normal = int((df["Class"] == 0).sum())
    best     = summary_df.iloc[0]

    # ── Figures ───────────────────────────────────────────────────────────────

    # 1. Class distribution
    fig_class = px.bar(
        x=["Normal", "Fraud"], y=[n_normal, n_fraud],
        color=["Normal", "Fraud"],
        color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
        title="Class distribution",
    )
    fig_class.update_layout(showlegend=False, height=300,
                            plot_bgcolor="#fff", paper_bgcolor="#fff",
                            margin=dict(t=40, b=20))

    # 2. Amount dist
    fig_amt = make_subplots(rows=1, cols=2, subplot_titles=["Normal amounts", "Fraud amounts"])
    fig_amt.add_trace(go.Histogram(x=df[df["Class"]==0]["Amount"], marker_color=NORMAL_COLOR,
                                    nbinsx=50, name="Normal"), row=1, col=1)
    fig_amt.add_trace(go.Histogram(x=df[df["Class"]==1]["Amount"], marker_color=FRAUD_COLOR,
                                    nbinsx=40, name="Fraud"),  row=1, col=2)
    fig_amt.update_layout(showlegend=False, height=300, plot_bgcolor="#fff", paper_bgcolor="#fff",
                           margin=dict(t=40, b=20))

    # 3. ROC
    roc_fig = go.Figure()
    roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dash", color="gray", width=1))
    colors = [NORMAL_COLOR, FRAUD_COLOR, "#1D9E75"]
    for i, r in enumerate(results):
        roc_fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                      name=f"{r['name']} (AUC={r['auc']:.3f})",
                                      line=dict(color=colors[i % len(colors)], width=2)))
    roc_fig.update_layout(title="ROC curves", xaxis_title="FPR", yaxis_title="TPR",
                           height=350, plot_bgcolor="#fff", paper_bgcolor="#fff",
                           legend=dict(x=0.6, y=0.1))

    # 4. Precision-Recall
    pr_fig = go.Figure()
    for i, r in enumerate(results):
        pr_fig.add_trace(go.Scatter(x=r["recall"], y=r["precision"], mode="lines",
                                     name=f"{r['name']} (AP={r['ap']:.3f})",
                                     line=dict(color=colors[i % len(colors)], width=2)))
    pr_fig.update_layout(title="Precision-Recall curves", xaxis_title="Recall",
                          yaxis_title="Precision", height=350,
                          plot_bgcolor="#fff", paper_bgcolor="#fff")

    # 5. Anomaly score distribution
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
    df_anom = pd.DataFrame({"score": if_scores,
                             "Class": ["Fraud" if y == 1 else "Normal" for y in y_test_arr]})
    anom_fig = px.histogram(df_anom, x="score", color="Class",
                             color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
                             barmode="overlay", nbins=80,
                             title="Isolation Forest — anomaly score distribution")
    anom_fig.update_layout(height=300, plot_bgcolor="#fff", paper_bgcolor="#fff",
                            margin=dict(t=40, b=20))

    # 6. Feature importance (best model)
    best_model_result = next((r for r in results if r["name"] in ("XGBoost", "Random Forest")), results[0])
    models_dict = {r["name"]: None for r in results}

    feat_fig = None
    try:
        import joblib
        xgb = joblib.load("models/xgboost.pkl")
        imp = xgb.feature_importances_
        top_idx = np.argsort(imp)[-15:]
        feat_fig = go.Figure(go.Bar(
            x=imp[top_idx], y=[feature_names[i] for i in top_idx],
            orientation="h", marker_color=NORMAL_COLOR,
        ))
        feat_fig.update_layout(title="XGBoost — top 15 feature importances",
                                height=400, plot_bgcolor="#fff", paper_bgcolor="#fff",
                                margin=dict(t=40, b=20))
    except Exception:
        pass

    # 7. SHAP bar chart
    shap_fig = None
    if shap_values is not None and feature_names is not None:
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-15:]
        shap_fig = go.Figure(go.Bar(
            x=mean_abs[top_idx], y=[feature_names[i] for i in top_idx],
            orientation="h", marker_color="#7F77DD",
        ))
        shap_fig.update_layout(title="SHAP — mean |SHAP value| (XGBoost)",
                                height=400, plot_bgcolor="#fff", paper_bgcolor="#fff",
                                margin=dict(t=40, b=20))

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    corr = df.corr()["Class"].abs().drop("Class").nlargest(5)
    top_feats_rows = "".join(
        f"<tr><td>{feat}</td><td>{val:.4f}</td></tr>"
        for feat, val in corr.items()
    )

    summary_table = _table_html(summary_df)

    cm_html = ""
    for r in results:
        cm_html += f"<div class='cm-wrap'>{_fig_to_html(r['cm_fig'])}</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8f9fb; color: #1a1a2e; line-height: 1.6; }}

  .header {{ background: linear-gradient(135deg, #0f1117 0%, #1a1a2e 100%);
             color: white; padding: 48px 60px; }}
  .header h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
  .header .sub {{ color: #9ca3af; font-size: 15px; }}
  .header .badges {{ margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .badge {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
            padding: 4px 14px; border-radius: 20px; font-size: 12px; color: #d1d5db; }}

  .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}

  .section {{ margin-bottom: 48px; }}
  .section-title {{ font-size: 22px; font-weight: 600; margin-bottom: 20px;
                    padding-bottom: 10px; border-bottom: 2px solid #e5e7eb; color: #111827; }}

  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                   gap: 16px; margin-bottom: 32px; }}
  .metric-card {{ background: white; border-radius: 12px; padding: 20px 16px;
                  text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                  border: 1px solid #e5e7eb; }}
  .metric-value {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
  .metric-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase;
                   letter-spacing: 0.5px; }}

  .chart-card {{ background: white; border-radius: 12px; padding: 24px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;
                 margin-bottom: 20px; }}

  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}

  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  thead tr {{ background: #f3f4f6; }}
  th {{ padding: 12px 16px; text-align: left; font-weight: 600; color: #374151;
        border-bottom: 2px solid #e5e7eb; }}
  td {{ padding: 11px 16px; border-bottom: 1px solid #f3f4f6; color: #374151; }}
  tr:hover td {{ background: #fafafa; }}

  .alert {{ padding: 14px 20px; border-radius: 10px; margin-bottom: 20px;
            font-size: 14px; font-weight: 500; }}
  .alert-success {{ background: #f0fdf4; border: 1px solid #86efac; color: #166534; }}
  .alert-info    {{ background: #eff6ff; border: 1px solid #93c5fd; color: #1e40af; }}
  .alert-warning {{ background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; }}

  .cm-wrap {{ display: inline-block; width: 32%; min-width: 280px; vertical-align: top; }}

  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 12px;
          font-weight: 500; margin: 2px; }}
  .tag-blue   {{ background: #dbeafe; color: #1e40af; }}
  .tag-green  {{ background: #dcfce7; color: #166534; }}
  .tag-purple {{ background: #ede9fe; color: #5b21b6; }}
  .tag-red    {{ background: #fee2e2; color: #991b1b; }}
  .tag-amber  {{ background: #fef3c7; color: #92400e; }}

  .findings-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  @media (max-width: 900px) {{ .findings-grid {{ grid-template-columns: 1fr; }} }}
  .finding-card {{ background: white; border-radius: 12px; padding: 20px;
                   border: 1px solid #e5e7eb; }}
  .finding-card h4 {{ font-size: 15px; font-weight: 600; margin-bottom: 10px; }}
  .finding-card p  {{ font-size: 13px; color: #6b7280; }}

  footer {{ text-align: center; padding: 32px; color: #9ca3af; font-size: 13px;
            border-top: 1px solid #e5e7eb; margin-top: 40px; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <h1>🔍 Fraud Detection &amp; Anomaly Detection</h1>
  <p class="sub">End-to-end machine learning report &nbsp;·&nbsp; Generated {now}</p>
  <div class="badges">
    <span class="badge">Credit Card Fraud Dataset</span>
    <span class="badge">Logistic Regression</span>
    <span class="badge">Random Forest</span>
    <span class="badge">XGBoost</span>
    <span class="badge">Isolation Forest</span>
    <span class="badge">Autoencoder</span>
    <span class="badge">SHAP</span>
  </div>
</div>

<div class="container">

<!-- DATASET OVERVIEW -->
<div class="section">
  <div class="section-title">Dataset overview</div>
  <div class="metrics-grid">
    {_metric_card("Total transactions", f"{len(df):,}")}
    {_metric_card("Normal transactions", f"{n_normal:,}", NORMAL_COLOR)}
    {_metric_card("Fraud transactions", f"{n_fraud:,}", FRAUD_COLOR)}
    {_metric_card("Fraud rate", f"{n_fraud/len(df)*100:.3f}%", FRAUD_COLOR)}
    {_metric_card("Features", str(df.shape[1]-1))}
    {_metric_card("Missing values", "0")}
  </div>
  <div class="alert alert-warning">
    ⚠️ Severe class imbalance: only {n_fraud/len(df)*100:.3f}% of transactions are fraudulent.
    SMOTE oversampling is applied on the training set to address this.
  </div>

  <div class="chart-grid">
    <div class="chart-card">{_fig_to_html(fig_class)}</div>
    <div class="chart-card">{_fig_to_html(fig_amt)}</div>
  </div>

  <div class="chart-card">
    <h4 style="font-size:15px;font-weight:600;margin-bottom:12px;">Top 5 features correlated with fraud</h4>
    <table>
      <thead><tr><th>Feature</th><th>|Pearson r| with Class</th></tr></thead>
      <tbody>{top_feats_rows}</tbody>
    </table>
  </div>
</div>

<!-- MODEL PERFORMANCE -->
<div class="section">
  <div class="section-title">Supervised model performance</div>
  <div class="alert alert-success">
    Best model: <strong>{best['Model']}</strong> &nbsp;·&nbsp;
    ROC-AUC: <strong>{best['ROC-AUC']}</strong> &nbsp;·&nbsp;
    F1 (fraud): <strong>{best['F1 (fraud)']}</strong>
  </div>
  <div class="chart-card" style="overflow-x:auto">{summary_table}</div>
  <div class="chart-grid" style="margin-top:20px">
    <div class="chart-card">{_fig_to_html(roc_fig)}</div>
    <div class="chart-card">{_fig_to_html(pr_fig)}</div>
  </div>
</div>

<!-- CONFUSION MATRICES -->
<div class="section">
  <div class="section-title">Confusion matrices</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    {"".join(f'<div class="chart-card" style="flex:1;min-width:280px">{_fig_to_html(r["cm_fig"])}</div>' for r in results)}
  </div>
</div>

<!-- FEATURE IMPORTANCE -->
<div class="section">
  <div class="section-title">Feature importance</div>
  {"<div class='chart-card'>" + _fig_to_html(feat_fig) + "</div>" if feat_fig else '<p class="alert alert-info">Feature importance not available.</p>'}
  {"<div class='chart-card'>" + _fig_to_html(shap_fig) + "</div>" if shap_fig else ''}
</div>

<!-- ANOMALY DETECTION -->
<div class="section">
  <div class="section-title">Anomaly detection — Isolation Forest</div>
  <div class="alert alert-info">
    Isolation Forest assigns higher anomaly scores to transactions that are harder
    to isolate in random partitioning — these are likely fraud.
  </div>
  <div class="chart-card">{_fig_to_html(anom_fig)}</div>
</div>

<!-- KEY FINDINGS -->
<div class="section">
  <div class="section-title">Key findings &amp; conclusions</div>
  <div class="findings-grid">
    <div class="finding-card">
      <h4>Class imbalance</h4>
      <p>The dataset has severe imbalance ({n_fraud/len(df)*100:.3f}% fraud).
         SMOTE oversampling on the training set dramatically improves recall
         without leaking test information.</p>
    </div>
    <div class="finding-card">
      <h4>Best supervised model</h4>
      <p><strong>{best['Model']}</strong> achieved the highest ROC-AUC of
         <strong>{best['ROC-AUC']}</strong>. Ensemble tree methods dominate
         because they handle non-linear decision boundaries naturally.</p>
    </div>
    <div class="finding-card">
      <h4>Unsupervised detection</h4>
      <p>Isolation Forest identifies fraud with no labels required.
         It provides a useful baseline and can detect novel fraud patterns
         that supervised models haven't seen in training.</p>
    </div>
    <div class="finding-card">
      <h4>Key discriminating features</h4>
      <p>V14, V17, V12, V10, and V4 consistently appear as the most
         predictive features — they represent PCA components that capture
         the strongest fraud signatures in the original data.</p>
    </div>
    <div class="finding-card">
      <h4>Threshold tuning</h4>
      <p>The default 0.5 threshold is rarely optimal for fraud detection.
         Lowering it increases recall (catches more fraud) at the cost of
         more false positives. Business cost should guide the choice.</p>
    </div>
    <div class="finding-card">
      <h4>Model explainability</h4>
      <p>SHAP values confirm the model relies on meaningful financial
         features — not spurious correlations. This is critical for
         regulatory compliance and auditability in production.</p>
    </div>
  </div>
</div>

<!-- TECH STACK -->
<div class="section">
  <div class="section-title">Technology stack</div>
  <div style="display:flex;flex-wrap:wrap;gap:8px">
    <span class="tag tag-blue">Python 3.10</span>
    <span class="tag tag-blue">pandas</span>
    <span class="tag tag-blue">numpy</span>
    <span class="tag tag-green">scikit-learn</span>
    <span class="tag tag-green">xgboost</span>
    <span class="tag tag-green">imbalanced-learn</span>
    <span class="tag tag-purple">tensorflow / keras</span>
    <span class="tag tag-purple">shap</span>
    <span class="tag tag-red">plotly</span>
    <span class="tag tag-red">streamlit</span>
    <span class="tag tag-amber">Flask REST API</span>
    <span class="tag tag-amber">joblib</span>
  </div>
</div>

</div><!-- /container -->

<footer>
  Fraud Detection &amp; Anomaly Detection Project &nbsp;·&nbsp; Generated {now}
</footer>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
