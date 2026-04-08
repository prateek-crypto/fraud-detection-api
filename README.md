# Fraud Detection & Anomaly Detection

A complete, production-grade ML project on the Kaggle Credit Card Fraud dataset.

## Quick start

```bash
pip install -r requirements.txt
python data/generate_sample_data.py   # no Kaggle needed
python run_pipeline.py --generate      # run full pipeline
streamlit run app/streamlit_app.py     # launch dashboard
python api/fraud_api.py                # start REST API
```

## Project structure

```
fraud_detection/
├── data/
│   ├── creditcard.csv              ← Kaggle dataset (or use generator)
│   └── generate_sample_data.py     ← synthetic data generator
├── utils/
│   ├── preprocessing.py            ← scaling, SMOTE, splits
│   ├── evaluation.py               ← metrics + Plotly charts
│   ├── report_generator.py         ← standalone HTML report
│   └── drift_detection.py          ← PSI + KS drift monitoring
├── models/
│   ├── train_models.py             ← LR, Random Forest, XGBoost
│   └── anomaly_detection.py        ← Isolation Forest, DBSCAN, Autoencoder
├── app/
│   └── streamlit_app.py            ← 11-page interactive dashboard
├── api/
│   └── fraud_api.py                ← Flask REST API
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── run_pipeline.py                 ← one-click full pipeline
└── requirements.txt
```

## Dashboard pages (11)
Home | EDA | Train Models | Model Evaluation | Model Comparison |
Anomaly Detection | Live Prediction | SHAP | Drift Detection | API Tester | Export Report

## API endpoints
POST /predict | POST /predict/batch | GET /health | GET /stats | GET/POST /model/threshold

## Tech stack
pandas · numpy · scikit-learn · xgboost · imbalanced-learn · tensorflow · shap · plotly · streamlit · flask · scipy
