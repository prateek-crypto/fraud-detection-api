"""
streamlit_app.py  (v2 — full project)
--------------------------------------
Run:   streamlit run app/streamlit_app.py

Pages:
  Home | EDA | Train Models | Model Evaluation | Model Comparison |
  Anomaly Detection | Live Prediction | SHAP | Drift Detection | View Report
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from utils.preprocessing  import load_data, preprocess, preprocess_single
from utils.evaluation     import (
    evaluate_model, metrics_summary_df,
    plot_class_distribution, plot_amount_distribution,
    plot_correlation_heatmap, plot_roc_curves,
    plot_precision_recall, plot_confusion_matrix,
    plot_anomaly_scores,
)
from models.train_models      import train_all
from models.anomaly_detection import (
    train_isolation_forest, isolation_forest_scores,
    train_autoencoder, autoencoder_scores,
    run_dbscan,
)

st.set_page_config(page_title="Fraud Detector", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#0f1117; }
[data-testid="stSidebar"] * { color:#fafafa !important; }
.mc  { background:#1e2130; border-radius:12px; padding:18px 22px; text-align:center; margin-bottom:8px; }
.mcv { font-size:28px; font-weight:600; color:#378ADD; }
.mcl { font-size:12px; color:#888; margin-top:4px; text-transform:uppercase; letter-spacing:.5px; }
.fraud-box  { background:#fee2e2; border:1px solid #fca5a5; border-radius:10px;
              padding:12px 18px; color:#991b1b; font-weight:600; }
.safe-box   { background:#dcfce7; border:1px solid #86efac; border-radius:10px;
              padding:12px 18px; color:#166534; font-weight:600; }
</style>""", unsafe_allow_html=True)

DATA_PATH = "data/creditcard.csv"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fraud Detector")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home", "📊 EDA", "🤖 Train Models", "📈 Model Evaluation",
        "⚖️ Model Comparison", "🚨 Anomaly Detection", "🔮 Live Prediction",
        "💡 SHAP Explainability", "📡 Drift Detection",
        "📄 View Report",
    ])
    st.markdown("---")
    st.markdown("**Dataset:** ULB Credit Card Fraud  \n**Rows:** 284,807  \n**Fraud:** 0.17%")
    st.success("✅ Dataset ready") if os.path.exists(DATA_PATH) else st.warning("⚠️ No dataset")

# ── Helpers ───────────────────────────────────────────────────────────────────
def mc(label, value, color="#378ADD"):
    return f'<div class="mc"><div class="mcv" style="color:{color}">{value}</div><div class="mcl">{label}</div></div>'

def get_data():
    if "df" not in st.session_state:
        if os.path.exists(DATA_PATH):
            st.session_state.df = load_data(DATA_PATH)
        else: return None
    return st.session_state.df

def get_splits():
    if "splits" not in st.session_state:
        df = get_data()
        if df is None: return None
        with st.spinner("Preprocessing…"):
            st.session_state.splits = preprocess(df)
    return st.session_state.splits

def get_models():
    if "models" not in st.session_state:
        s = get_splits()
        if s is None: return None
        X_train, _, y_train, _, _ = s
        with st.spinner("Training (~2 min)…"):
            st.session_state.models = train_all(X_train, y_train, verbose=False)
    return st.session_state.models

def get_results():
    if "eval_results" not in st.session_state:
        m = get_models(); s = get_splits()
        if m is None: return None
        _, X_test, _, y_test, _ = s
        st.session_state.eval_results = [evaluate_model(n, mdl, X_test, y_test) for n, mdl in m.items()]
    return st.session_state.eval_results

def no_data():
    st.warning("Dataset not found. Place `creditcard.csv` in `data/`, or generate synthetic data:")
    if st.button("Generate 100k synthetic transactions"):
        from data.generate_sample_data import generate_creditcard_like
        with st.spinner("Generating…"):
            df = generate_creditcard_like(100_000)
            os.makedirs("data", exist_ok=True); df.to_csv(DATA_PATH, index=False)
            st.session_state.df = df
        st.success("Done! Refresh the page."); st.rerun()

def no_model(): st.warning("Train models first (🤖 Train Models page).")

# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🔍 Fraud Detection & Anomaly Detection")
    st.markdown("An end-to-end ML project — EDA → models → anomaly detection → explainability → production.")
    st.markdown("---")
    cols = st.columns(4)
    for col, (label, val, color) in zip(cols, [
        ("Transactions", "284K", "#378ADD"), ("Fraud cases", "492", "#E24B4A"),
        ("Fraud rate",   "0.17%", "#EF9F27"), ("Features",   "30", "#1D9E75"),
    ]):
        col.markdown(mc(label, val, color), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Supervised ML\nLogistic Regression · Random Forest · XGBoost\n\n"
                    "### Preprocessing\nStandard scaling · SMOTE · 80/20 stratified split")
    with c2:
        st.markdown("### Anomaly Detection\nIsolation Forest · DBSCAN + t-SNE · Autoencoder\n\n"
                    "### Production-ready\nSHAP explainability · Drift detection · REST API · HTML report")
    if not os.path.exists(DATA_PATH):
        st.markdown("---"); no_data()
    else:
        st.success("✅ Dataset found. Use the sidebar to start exploring.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    df = get_data()
    if df is None: no_data(); st.stop()

    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}"); c2.metric("Fraud", f"{df['Class'].sum():,}")
    c3.metric("Missing", str(df.isnull().sum().sum()))
    st.markdown("---")
    st.plotly_chart(plot_class_distribution(df), use_container_width=True)
    st.plotly_chart(plot_amount_distribution(df), use_container_width=True)
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    st.subheader("Box plots — top 6 features")
    top_feats = df.corr()["Class"].abs().drop("Class").nlargest(6).index.tolist()
    cols = st.columns(3)
    for i, feat in enumerate(top_feats):
        fig = px.box(df, x=df["Class"].map({0: "Normal", 1: "Fraud"}), y=feat,
                     color=df["Class"].map({0: "Normal", 1: "Fraud"}),
                     color_discrete_map={"Normal": "#378ADD", "Fraud": "#E24B4A"},
                     title=feat, labels={"x": "Class"})
        fig.update_layout(showlegend=False, plot_bgcolor="#fff", paper_bgcolor="#fff", margin=dict(t=40, b=20))
        cols[i % 3].plotly_chart(fig, use_container_width=True)
    st.dataframe(df.describe().T.round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.title("🤖 Train Supervised Models")
    df = get_data()
    if df is None: no_data(); st.stop()

    if st.button("▶ Preprocess + Train all models", type="primary"):
        s = get_splits()
        X_train, X_test, y_train, y_test, _ = s
        st.success(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
        st.info(f"After SMOTE — fraud in train: {y_train.sum():,}/{len(y_train):,}")
        m = get_models()
        for name in m: st.success(f"✅ {name} trained & saved to models/")
        st.balloons()
    else:
        if st.session_state.get("models"):
            st.success("✅ Models ready — go to Model Evaluation.")
        else:
            st.info("Click above to begin training.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    results = get_results()
    if results is None: no_model(); st.stop()
    splits = get_splits(); _, X_test, _, y_test, _ = splits

    st.subheader("Performance summary")
    st.dataframe(metrics_summary_df(results).style.highlight_max(axis=0, color="#d4f0c0"),
                 use_container_width=True)
    st.plotly_chart(plot_roc_curves({r["name"]: (r["fpr"], r["tpr"], r["auc"]) for r in results}),
                    use_container_width=True)
    st.plotly_chart(plot_precision_recall({r["name"]: (r["precision"], r["recall"], r["ap"]) for r in results}),
                    use_container_width=True)
    cols = st.columns(len(results))
    for i, r in enumerate(results): cols[i].plotly_chart(r["cm_fig"], use_container_width=True)

    st.subheader("Threshold tuning — XGBoost")
    xgb_r = next((r for r in results if "XGBoost" in r["name"]), None)
    if xgb_r:
        t = st.slider("Threshold", 0.01, 0.99, 0.5, 0.01)
        from sklearn.metrics import precision_score, recall_score, f1_score
        yp = (xgb_r["y_prob"] >= t).astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision_score(y_test, yp, zero_division=0):.4f}")
        c2.metric("Recall",    f"{recall_score(y_test, yp, zero_division=0):.4f}")
        c3.metric("F1",        f"{f1_score(y_test, yp, zero_division=0):.4f}")
        st.plotly_chart(plot_confusion_matrix(y_test, yp, f"XGBoost @ {t}"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Model Comparison":
    st.title("⚖️ Deep Model Comparison")
    results = get_results()
    if results is None: no_model(); st.stop()
    splits = get_splits(); _, X_test, _, y_test, feat = splits

    # Radar
    metrics  = ["ROC-AUC", "Avg Precision", "F1 (fraud)", "Precision (fraud)", "Recall (fraud)"]
    summary  = metrics_summary_df(results)
    colors   = ["#378ADD", "#E24B4A", "#1D9E75"]
    fig_rad  = go.Figure()
    for i, (_, row) in enumerate(summary.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        fig_rad.add_trace(go.Scatterpolar(r=vals, theta=metrics+[metrics[0]],
                                           fill="toself", name=row["Model"],
                                           line_color=colors[i%3], opacity=0.75))
    fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                           title="Radar — multi-metric", height=450,
                           plot_bgcolor="#fff", paper_bgcolor="#fff")
    st.plotly_chart(fig_rad, use_container_width=True)

    # Score distributions
    st.subheader("Fraud probability distributions")
    cols = st.columns(len(results))
    for i, r in enumerate(results):
        df_d = pd.DataFrame({"prob": r["y_prob"], "true": y_test.values})
        fig  = px.histogram(df_d, x="prob",
                             color=df_d["true"].map({0:"Normal",1:"Fraud"}),
                             color_discrete_map={"Normal":"#378ADD","Fraud":"#E24B4A"},
                             barmode="overlay", nbins=60, title=r["name"], opacity=0.7)
        fig.update_layout(showlegend=False, plot_bgcolor="#fff", paper_bgcolor="#fff",
                          margin=dict(t=40,b=10), height=250)
        cols[i].plotly_chart(fig, use_container_width=True)

    # Calibration
    st.subheader("Probability calibration curves")
    from sklearn.calibration import calibration_curve
    fig_cal = go.Figure()
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    for i, r in enumerate(results):
        fp, mp = calibration_curve(y_test, r["y_prob"], n_bins=10)
        fig_cal.add_trace(go.Scatter(x=mp, y=fp, mode="lines+markers",
                                      name=r["name"], line=dict(color=colors[i%3], width=2)))
    fig_cal.update_layout(title="Calibration (closer to diagonal = better)",
                           xaxis_title="Mean predicted probability", yaxis_title="Fraction positives",
                           plot_bgcolor="#fff", paper_bgcolor="#fff", height=400)
    st.plotly_chart(fig_cal, use_container_width=True)

    # Error analysis
    st.subheader("Error analysis")
    sel = st.selectbox("Model", [r["name"] for r in results])
    r   = next(r for r in results if r["name"] == sel)
    X_te_df = X_test.copy()
    X_te_df["y_true"] = y_test.values
    X_te_df["y_pred"] = r["y_pred"]
    X_te_df["y_prob"] = r["y_prob"]
    fn = X_te_df[(X_te_df["y_true"]==1) & (X_te_df["y_pred"]==0)]
    fp = X_te_df[(X_te_df["y_true"]==0) & (X_te_df["y_pred"]==1)]
    c1, c2 = st.columns(2)
    c1.metric("False Negatives (missed fraud)", len(fn))
    c2.metric("False Positives (wrong alarm)",  len(fp))
    if len(fn) > 0:
        st.markdown("**Top false negatives — highest-confidence misses**")
        st.dataframe(fn.sort_values("y_prob", ascending=False).head(10)
                     [["y_prob","y_true","y_pred"]+feat[:6]], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    splits = get_splits()
    if splits is None: no_model(); st.stop()
    X_train, X_test, y_train, y_test, _ = splits
    X_tr = X_train.values; X_te = X_test.values; y_tr = y_train.values

    tabs = st.tabs(["Isolation Forest", "DBSCAN + t-SNE", "Autoencoder"])

    with tabs[0]:
        st.subheader("Isolation Forest")
        st.markdown("Anomalies = transactions that are easy to isolate in random partitioning.")
        contamination = st.slider("Contamination", 0.001, 0.01, 0.002, 0.001)
        if st.button("Train Isolation Forest", type="primary"):
            with st.spinner("Training…"):
                st.session_state.ifo = train_isolation_forest(X_tr, contamination=contamination)
            st.success("Done!")
        ifo = st.session_state.get("ifo")
        if ifo:
            scores = isolation_forest_scores(ifo, X_te)
            st.plotly_chart(plot_anomaly_scores(scores, y_test), use_container_width=True)
            t = st.slider("Threshold", 0.3, 0.99, 0.7, 0.01, key="if_t")
            yp = (scores >= t).astype(int)
            from sklearn.metrics import classification_report
            rep = classification_report(y_test, yp, output_dict=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{rep['1']['precision']:.4f}")
            c2.metric("Recall",    f"{rep['1']['recall']:.4f}")
            c3.metric("F1",        f"{rep['1']['f1-score']:.4f}")
            st.plotly_chart(plot_confusion_matrix(y_test, yp, "Isolation Forest"), use_container_width=True)

    with tabs[1]:
        st.subheader("DBSCAN + t-SNE")
        n = st.slider("Sample size", 2000, 15000, 5000, 1000)
        eps = st.slider("eps", 0.1, 5.0, 1.5, 0.1)
        ms  = st.slider("min_samples", 3, 30, 10)
        if st.button("Run DBSCAN + t-SNE", type="primary"):
            with st.spinner("Running (~1-2 min)…"):
                labels, X2d, idx = run_dbscan(X_te, eps=eps, min_samples=ms, sample_size=n)
                y_s = y_test.iloc[idx]
                st.session_state.update({"dbscan_labels": labels, "dbscan_X2d": X2d, "dbscan_y": y_s})
        if "dbscan_labels" in st.session_state:
            labels = st.session_state.dbscan_labels
            X2d    = st.session_state.dbscan_X2d
            y_s    = st.session_state.dbscan_y
            noise  = labels == -1
            df_p   = pd.DataFrame({"x": X2d[:,0], "y": X2d[:,1],
                                    "status": ["True fraud" if t==1 else ("Anomaly" if n else "Normal")
                                               for t, n in zip(y_s.values, noise)]})
            fig = px.scatter(df_p, x="x", y="y", color="status", opacity=0.65,
                             color_discrete_map={"True fraud":"#E24B4A","Anomaly":"#EF9F27","Normal":"#378ADD"},
                             title="DBSCAN — t-SNE projection")
            fig.update_traces(marker_size=4)
            fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff")
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("DBSCAN noise points", int(noise.sum()))
            c2.metric("True fraud in noise", int((y_s.values[noise]==1).sum()))

    with tabs[2]:
        st.subheader("Autoencoder (Keras)")
        st.markdown("Trained on normal transactions only — fraud has higher reconstruction error.")
        epochs = st.slider("Epochs", 5, 50, 20)
        if st.button("Train Autoencoder", type="primary"):
            X_norm = X_tr[y_tr == 0]; split = int(0.9 * len(X_norm))
            with st.spinner("Training…"):
                ae, ae_sc, hist = train_autoencoder(X_norm[:split], X_norm[split:], epochs=epochs)
                st.session_state.ae = ae; st.session_state.ae_sc = ae_sc
            st.success("Done!")
            ld = pd.DataFrame({"epoch": range(1,len(hist.history["loss"])+1),
                                "train": hist.history["loss"], "val": hist.history["val_loss"]})
            fig = px.line(ld, x="epoch", y=["train","val"], title="Autoencoder training loss")
            fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff")
            st.plotly_chart(fig, use_container_width=True)
        ae = st.session_state.get("ae"); ae_sc = st.session_state.get("ae_sc")
        if ae and ae_sc:
            scores = autoencoder_scores(ae, ae_sc, X_te)
            st.plotly_chart(plot_anomaly_scores(scores, y_test, "Autoencoder (MSE)"), use_container_width=True)
            t = st.slider("Threshold", float(np.percentile(scores,80)),
                          float(np.percentile(scores,99.9)), float(np.percentile(scores,95)), key="ae_t")
            yp = (scores >= t).astype(int)
            from sklearn.metrics import classification_report
            rep = classification_report(y_test, yp, output_dict=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{rep['1']['precision']:.4f}")
            c2.metric("Recall",    f"{rep['1']['recall']:.4f}")
            c3.metric("F1",        f"{rep['1']['f1-score']:.4f}")
            st.plotly_chart(plot_confusion_matrix(y_test, yp, "Autoencoder"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Fraud Prediction")
    m = get_models()
    if m is None: no_model(); st.stop()
    model_name = st.selectbox("Model", list(m.keys()))
    model      = m[model_name]
    tab1, tab2 = st.tabs(["Upload CSV", "Manual entry"])

    with tab1:
        f = st.file_uploader("Upload transactions CSV", type="csv")
        if f:
            df_up = pd.read_csv(f)
            st.dataframe(df_up.head(), use_container_width=True)
            if st.button("Predict", type="primary"):
                try:
                    X_up = preprocess_single(df_up)
                    probs = model.predict_proba(X_up)[:, 1]
                    df_up["fraud_prob"] = probs.round(4)
                    df_up["label"] = (probs >= 0.5).map({True:"🚨 FRAUD", False:"✅ Normal"})
                    st.dataframe(df_up[["fraud_prob","label"]], use_container_width=True)
                    fn = (probs >= 0.5).sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total", len(df_up)); c2.metric("Fraud flagged", fn)
                    c3.metric("Fraud rate", f"{fn/len(df_up)*100:.2f}%")
                    fig = px.histogram(df_up, x="fraud_prob", nbins=50, title="Fraud prob distribution",
                                       color_discrete_sequence=["#378ADD"])
                    fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff")
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("Download predictions", df_up.to_csv(index=False).encode(),
                                       "predictions.csv", "text/csv")
                except Exception as e: st.error(str(e))

    with tab2:
        splits = get_splits(); _, X_test, _, _, feat = splits
        defaults = X_test.iloc[0].to_dict()
        cols = st.columns(3)
        inputs = {f: cols[i%3].number_input(f, value=float(defaults[f]), format="%.4f")
                  for i, f in enumerate(feat)}
        if st.button("Predict transaction", type="primary"):
            prob = model.predict_proba(pd.DataFrame([inputs]))[0][1]
            if prob >= 0.5:
                st.markdown(f'<div class="fraud-box">🚨 FRAUD — confidence: {prob*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-box">✅ Normal — fraud probability: {prob*100:.2f}%</div>', unsafe_allow_html=True)
            st.progress(float(prob))

# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 SHAP Explainability":
    st.title("💡 SHAP Explainability")
    results = get_results(); splits = get_splits()
    if results is None: no_model(); st.stop()
    _, X_test, _, y_test, feat = splits
    m = get_models()

    model_name = st.selectbox("Model", ["Random Forest", "XGBoost"])
    n_samples  = st.slider("Samples for SHAP", 100, 1000, 200, 50)

    if st.button("Compute SHAP values", type="primary"):
        import shap
        model = m[model_name]; X_s = X_test.iloc[:n_samples]
        with st.spinner("Computing…"):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_s)
            if isinstance(shap_values, list): shap_values = shap_values[1]
            st.session_state.update({"shap_values": shap_values, "shap_X": X_s, "shap_exp": explainer})
        st.success("Done!")

    if "shap_values" in st.session_state:
        import shap, matplotlib.pyplot as plt
        sv = st.session_state.shap_values; X_s = st.session_state.shap_X
        mean_shap = np.abs(sv).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[-20:]
        fig = go.Figure(go.Bar(x=mean_shap[top_idx], y=[feat[i] for i in top_idx],
                                orientation="h", marker_color="#378ADD"))
        fig.update_layout(title="Top 20 — mean |SHAP|", plot_bgcolor="#fff",
                          paper_bgcolor="#fff", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Beeswarm summary")
        shap.summary_plot(sv, X_s, feature_names=feat, show=False, max_display=15)
        st.pyplot(plt.gcf()); plt.clf()

        st.subheader("Waterfall — single transaction")
        idx = st.slider("Transaction index", 0, len(X_s)-1, 0)
        exp = st.session_state.shap_exp
        base = exp.expected_value if not isinstance(exp.expected_value, list) else exp.expected_value[1]
        shap.waterfall_plot(
            shap.Explanation(values=sv[idx], base_values=base,
                             data=X_s.iloc[idx].values, feature_names=feat),
            show=False, max_display=15)
        st.pyplot(plt.gcf()); plt.clf()

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Drift Detection":
    st.title("📡 Data Drift Detection")
    splits = get_splits()
    if splits is None: no_model(); st.stop()
    X_train, X_test, _, _, feat = splits

    from utils.drift_detection import DriftDetector

    mode = st.radio("New data source", [
        "Use test set (minimal drift expected)",
        "Simulate artificial drift",
        "Upload new CSV",
    ])
    X_new = None
    if mode == "Use test set (minimal drift expected)":
        X_new = X_test
    elif mode == "Simulate artificial drift":
        shift = st.slider("Shift magnitude", 0.0, 5.0, 2.0, 0.1)
        n_sh  = st.slider("Features to shift", 1, 10, 5)
        X_new = X_test.copy()
        scols = np.random.choice(feat, n_sh, replace=False)
        for c in scols: X_new[c] += shift
        st.info(f"Shifted: {', '.join(scols)}")
    else:
        f = st.file_uploader("Upload new CSV", type="csv")
        if f:
            df_up = pd.read_csv(f)
            X_new = preprocess_single(df_up) if "Amount" in df_up.columns else df_up

    if X_new is not None and st.button("Run Drift Detection", type="primary"):
        with st.spinner("Analysing…"):
            detector = DriftDetector(X_train)
            report   = detector.check(X_new)
        emoji = "🔴" if report["n_critical"] > 0 else "🟡" if report["n_warning"] > 0 else "🟢"
        st.markdown(f"## {emoji} {report['recommendation']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("OK",       report["n_ok"])
        c2.metric("Warning",  report["n_warning"])
        c3.metric("Critical", report["n_critical"])
        st.plotly_chart(detector.plot_drift_report(report), use_container_width=True)
        st.subheader("Per-feature detail")
        df_s = report["feature_stats"]
        st.dataframe(df_s.style.apply(
            lambda col: ["background:#fee2e2" if v=="CRITICAL" else "background:#fffbeb" if v=="WARNING" else "" for v in col],
            subset=["Drift"]), use_container_width=True)
        sel = st.selectbox("Feature distribution comparison", feat)
        st.plotly_chart(detector.plot_feature_comparison(X_new, sel), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════

elif page == "📄 View Report":
    st.title("📄 Fraud Detection Report")
    results = get_results(); splits = get_splits(); df = get_data()
    if results is None or splits is None or df is None: no_model(); st.stop()

    _, X_test, _, y_test, feat = splits
    ifo = st.session_state.get("ifo")
    has_ifo  = ifo is not None
    has_shap = "shap_values" in st.session_state

    c1, c2, c3 = st.columns(3)
    c1.metric("Models evaluated",  len(results))
    c2.metric("Isolation Forest", "✅" if has_ifo  else "❌ (optional)")
    c3.metric("SHAP values",      "✅" if has_shap else "❌ (optional)")
    st.info("For the richest report, run Anomaly Detection + SHAP pages first.")

    if st.button("Generate and View Report", type="primary"):
        from utils.report_generator import generate_report
        from utils.evaluation import metrics_summary_df
        if_scores = isolation_forest_scores(ifo, X_test.values) if ifo else np.zeros(len(X_test))
        with st.spinner("Building report…"):
            path = generate_report(
                df=df, results=results, summary_df=metrics_summary_df(results),
                if_scores=if_scores, y_test=y_test,
                shap_values=st.session_state.get("shap_values"),
                shap_X=st.session_state.get("shap_X"),
                feature_names=feat,
            )
        with open(path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.success(f"✅ Report generated at `{path}`")
        st.components.v1.html(html_content, height=800, scrolling=True)
