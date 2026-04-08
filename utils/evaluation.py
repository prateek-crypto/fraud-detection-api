import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)

# ── palette ──────────────────────────────────────────────────────────────────
FRAUD_COLOR  = "#E24B4A"
NORMAL_COLOR = "#378ADD"
BG           = "#FFFFFF"

def plot_class_distribution(df, return_fig=True):
    counts = df["Class"].value_counts().sort_index()
    labels = ["Normal (0)", "Fraud (1)"]
    colors = [NORMAL_COLOR, FRAUD_COLOR]
    fig = px.bar(
        x=labels, y=counts.values, color=labels,
        color_discrete_sequence=colors,
        title="Class distribution",
        labels={"x": "Class", "y": "Count"},
    )
    fig.update_layout(showlegend=False, plot_bgcolor=BG, paper_bgcolor=BG)
    return fig

def plot_amount_distribution(df, return_fig=True):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Normal transactions", "Fraud transactions"])
    fig.add_trace(go.Histogram(x=df[df["Class"]==0]["Amount"], marker_color=NORMAL_COLOR,
                               name="Normal", nbinsx=50, opacity=0.75), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[df["Class"]==1]["Amount"], marker_color=FRAUD_COLOR,
                               name="Fraud",  nbinsx=50, opacity=0.75), row=1, col=2)
    fig.update_layout(title="Transaction amount distribution", showlegend=False,
                      plot_bgcolor=BG, paper_bgcolor=BG)
    return fig

def plot_correlation_heatmap(df, return_fig=True):
    corr = df.corr()["Class"].drop("Class").sort_values()
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=[FRAUD_COLOR if v > 0 else NORMAL_COLOR for v in corr.values],
    ))
    fig.update_layout(title="Feature correlation with Class label",
                      xaxis_title="Pearson correlation",
                      plot_bgcolor=BG, paper_bgcolor=BG, height=600)
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        color_continuous_scale=["#E6F1FB", "#185FA5"],
        labels=dict(x="Predicted", y="Actual"),
        x=["Normal", "Fraud"], y=["Normal", "Fraud"],
        title=f"Confusion matrix — {model_name}",
    )
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
    return fig

def plot_roc_curves(results: dict):
    """results = {model_name: (fpr, tpr, auc)}"""
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray", width=1))
    colors = ["#378ADD", "#E24B4A", "#1D9E75", "#EF9F27", "#7F77DD"]
    for i, (name, (fpr, tpr, auc)) in enumerate(results.items()):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.3f})",
                                 line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(
        title="ROC curves — all models",
        xaxis_title="False positive rate", yaxis_title="True positive rate",
        plot_bgcolor=BG, paper_bgcolor=BG, legend=dict(x=0.6, y=0.1),
    )
    return fig

def plot_precision_recall(results_pr: dict):
    """results_pr = {model_name: (precision, recall, ap)}"""
    fig = go.Figure()
    colors = ["#378ADD", "#E24B4A", "#1D9E75", "#EF9F27", "#7F77DD"]
    for i, (name, (prec, rec, ap)) in enumerate(results_pr.items()):
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{name} (AP={ap:.3f})",
                                 line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(
        title="Precision-recall curves — all models",
        xaxis_title="Recall", yaxis_title="Precision",
        plot_bgcolor=BG, paper_bgcolor=BG,
    )
    return fig

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """
    Returns a dict with all metrics and plotly figures.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc          = roc_auc_score(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap           = average_precision_score(y_test, y_prob)
    f1           = f1_score(y_test, y_pred)
    report       = classification_report(y_test, y_pred, output_dict=True)

    return {
        "name": name, "y_prob": y_prob, "y_pred": y_pred,
        "auc": auc, "ap": ap, "f1": f1,
        "fpr": fpr, "tpr": tpr,
        "precision": prec, "recall": rec,
        "report": report,
        "cm_fig": plot_confusion_matrix(y_test, y_pred, name),
    }

def plot_feature_importance(feature_names, importances, model_name="Model", top_n=20):
    idx   = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx], y=[feature_names[i] for i in idx],
        orientation="h", marker_color=NORMAL_COLOR,
    ))
    fig.update_layout(
        title=f"Top {top_n} feature importances — {model_name}",
        xaxis_title="Importance", plot_bgcolor=BG, paper_bgcolor=BG, height=500,
    )
    return fig

def plot_anomaly_scores(scores, y_true, model_name="Isolation Forest"):
    df = pd.DataFrame({"score": scores, "label": y_true.values})
    fig = px.histogram(df, x="score", color=df["label"].map({0: "Normal", 1: "Fraud"}),
                       color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
                       barmode="overlay", nbins=80,
                       title=f"Anomaly score distribution — {model_name}",
                       labels={"score": "Anomaly score", "color": "Class"})
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
    return fig

def plot_tsne(X_2d, y_true, title="t-SNE — fraud vs normal"):
    df = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1],
                       "label": y_true.map({0: "Normal", 1: "Fraud"})})
    fig = px.scatter(df, x="x", y="y", color="label",
                     color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
                     title=title, opacity=0.6,
                     labels={"x": "Component 1", "y": "Component 2"})
    fig.update_traces(marker_size=3)
    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
    return fig

def metrics_summary_df(eval_results: list) -> pd.DataFrame:
    rows = []
    for r in eval_results:
        rep = r["report"]
        rows.append({
            "Model": r["name"],
            "ROC-AUC": round(r["auc"], 4),
            "Avg Precision": round(r["ap"], 4),
            "F1 (fraud)": round(r["f1"], 4),
            "Precision (fraud)": round(rep["1"]["precision"], 4),
            "Recall (fraud)": round(rep["1"]["recall"], 4),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
