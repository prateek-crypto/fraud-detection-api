"""
utils/drift_detection.py
------------------------
Monitors for data drift and model performance degradation
between a reference (training) dataset and new incoming data.

Usage:
    from utils.drift_detection import DriftDetector
    detector = DriftDetector(X_reference, y_reference)
    report   = detector.check(X_new, y_new_optional)
    detector.plot_drift_report(report).show()
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats


class DriftDetector:
    """
    Detects statistical drift between a reference distribution and
    a new batch of data using KS test and Population Stability Index (PSI).
    """

    PSI_THRESHOLD_WARNING  = 0.1   # low change
    PSI_THRESHOLD_CRITICAL = 0.25  # significant change

    KS_PVALUE_THRESHOLD = 0.05     # reject H0: same distribution

    def __init__(self, X_reference: pd.DataFrame, y_reference=None):
        """
        X_reference: pandas DataFrame of features from training/baseline
        y_reference: optional series of labels
        """
        self.X_ref   = X_reference
        self.y_ref   = y_reference
        self.columns = X_reference.columns.tolist()

    # ── PSI ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index.
        PSI < 0.1  : no significant change
        PSI < 0.25 : moderate change — investigate
        PSI >= 0.25: significant change — retrain
        """
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0

        exp_counts = np.histogram(expected, bins=breakpoints)[0] + 1e-6
        act_counts = np.histogram(actual,   bins=breakpoints)[0] + 1e-6

        exp_pct = exp_counts / exp_counts.sum()
        act_pct = act_counts / act_counts.sum()

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct + 1e-9))
        return float(psi)

    # ── KS test ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ks(reference: np.ndarray, current: np.ndarray):
        """Two-sample Kolmogorov-Smirnov test."""
        stat, pval = stats.ks_2samp(reference, current)
        return float(stat), float(pval)

    # ── Check ─────────────────────────────────────────────────────────────────
    def check(self, X_new: pd.DataFrame, y_new=None) -> dict:
        """
        Run drift checks on all features.
        Returns a dict with per-feature stats and a summary.
        """
        rows = []
        for col in self.columns:
            if col not in X_new.columns:
                continue
            ref = self.X_ref[col].dropna().values
            cur = X_new[col].dropna().values

            psi         = self._psi(ref, cur)
            ks_stat, ks_pval = self._ks(ref, cur)

            if psi >= self.PSI_THRESHOLD_CRITICAL:
                severity = "CRITICAL"
            elif psi >= self.PSI_THRESHOLD_WARNING:
                severity = "WARNING"
            else:
                severity = "OK"

            rows.append({
                "Feature":    col,
                "PSI":        round(psi, 5),
                "KS stat":    round(ks_stat, 4),
                "KS p-value": round(ks_pval, 4),
                "Drift":      severity,
                "Ref mean":   round(float(ref.mean()), 4),
                "New mean":   round(float(cur.mean()), 4),
                "Mean shift": round(float(cur.mean() - ref.mean()), 4),
            })

        df = pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)

        n_critical = (df["Drift"] == "CRITICAL").sum()
        n_warning  = (df["Drift"] == "WARNING").sum()

        return {
            "feature_stats": df,
            "n_critical":    int(n_critical),
            "n_warning":     int(n_warning),
            "n_ok":          int(len(df) - n_critical - n_warning),
            "top_drifted":   df["Feature"].head(5).tolist(),
            "recommendation": (
                "RETRAIN — significant data drift detected across multiple features."
                if n_critical >= 3 else
                "MONITOR — moderate drift in some features. Check before next release."
                if n_warning >= 5 or n_critical >= 1 else
                "STABLE — no significant drift detected."
            ),
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    def plot_drift_report(self, report: dict) -> go.Figure:
        df = report["feature_stats"]

        color_map = {"OK": "#1D9E75", "WARNING": "#EF9F27", "CRITICAL": "#E24B4A"}
        colors = [color_map[d] for d in df["Drift"]]

        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "PSI by feature (sorted)", "KS statistic by feature",
            "Mean shift (new − reference)", "Drift severity summary",
        ])

        fig.add_trace(go.Bar(x=df["Feature"], y=df["PSI"], marker_color=colors,
                              name="PSI"), row=1, col=1)
        fig.add_hline(y=self.PSI_THRESHOLD_WARNING,  line_dash="dash",
                      line_color="#EF9F27", row=1, col=1)
        fig.add_hline(y=self.PSI_THRESHOLD_CRITICAL, line_dash="dash",
                      line_color="#E24B4A", row=1, col=1)

        fig.add_trace(go.Bar(x=df["Feature"], y=df["KS stat"], marker_color=colors,
                              name="KS stat"), row=1, col=2)
        fig.add_hline(y=0.1, line_dash="dash", line_color="gray", row=1, col=2)

        fig.add_trace(go.Bar(x=df["Feature"], y=df["Mean shift"],
                              marker_color=["#E24B4A" if v > 0 else "#378ADD"
                                            for v in df["Mean shift"]],
                              name="Mean shift"), row=2, col=1)

        counts = [report["n_ok"], report["n_warning"], report["n_critical"]]
        labels = ["OK", "Warning", "Critical"]
        pie_colors = ["#1D9E75", "#EF9F27", "#E24B4A"]
        fig.add_trace(go.Pie(labels=labels, values=counts,
                              marker_colors=pie_colors, hole=0.4,
                              name="Severity"), row=2, col=2)

        fig.update_layout(
            title=f"Data Drift Report — {report['recommendation']}",
            height=700, showlegend=False,
            plot_bgcolor="#fff", paper_bgcolor="#fff",
        )
        fig.update_xaxes(tickangle=45)
        return fig

    def plot_feature_comparison(self, X_new: pd.DataFrame, feature: str) -> go.Figure:
        """Side-by-side histogram comparing reference vs new data for one feature."""
        ref = self.X_ref[feature].values
        cur = X_new[feature].values

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ref, name="Reference", nbinsx=50,
                                    marker_color="#378ADD", opacity=0.65))
        fig.add_trace(go.Histogram(x=cur, name="New data",  nbinsx=50,
                                    marker_color="#E24B4A",  opacity=0.65))
        fig.update_layout(
            barmode="overlay",
            title=f"Distribution shift — {feature}",
            xaxis_title=feature, yaxis_title="Count",
            plot_bgcolor="#fff", paper_bgcolor="#fff",
        )
        return fig


# ── Model performance drift ───────────────────────────────────────────────────
class ModelPerformanceMonitor:
    """
    Tracks model performance metrics over time (rolling windows).
    """

    def __init__(self):
        self.records = []

    def log(self, timestamp: str, y_true, y_pred, y_prob=None):
        """Log a batch of predictions."""
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        record = {
            "timestamp":  timestamp,
            "n_samples":  len(y_true),
            "fraud_rate": float(np.mean(y_true)),
            "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        }
        if y_prob is not None:
            try:
                record["auc"] = round(roc_auc_score(y_true, y_prob), 4)
            except Exception:
                record["auc"] = None
        self.records.append(record)
        return record

    def plot_metrics_over_time(self) -> go.Figure:
        if not self.records:
            raise ValueError("No records logged yet.")
        df = pd.DataFrame(self.records)
        fig = go.Figure()
        colors = {"f1": "#378ADD", "precision": "#1D9E75",
                  "recall": "#E24B4A", "auc": "#7F77DD"}
        for metric, color in colors.items():
            if metric in df.columns:
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[metric],
                                          mode="lines+markers", name=metric.upper(),
                                          line=dict(color=color, width=2)))
        fig.update_layout(
            title="Model performance over time",
            xaxis_title="Batch", yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(x=0.01, y=0.01),
        )
        return fig

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
