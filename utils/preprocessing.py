import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud cases   : {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    print(f"Normal cases  : {(df['Class']==0).sum():,}")
    return df

def get_class_distribution(df: pd.DataFrame) -> dict:
    """Return class distribution as a dict."""
    return {
        "normal": int((df["Class"] == 0).sum()),
        "fraud": int((df["Class"] == 1).sum()),
        "fraud_pct": round(df["Class"].mean() * 100, 4),
    }

def preprocess(df: pd.DataFrame, save_scaler: bool = True, scaler_path: str = "models/scaler.pkl"):
    """
    Full preprocessing pipeline:
    - Scale 'Amount' and 'Time'
    - Drop raw columns
    - Train/test split
    - SMOTE oversampling on train set only
    Returns X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nBefore SMOTE — train fraud: {y_train.sum()} / {len(y_train)}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"After  SMOTE — train fraud: {y_train_res.sum()} / {len(y_train_res)}")

    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    return X_train_res, X_test, y_train_res, y_test, feature_names

def preprocess_single(df: pd.DataFrame, scaler_path: str = "models/scaler.pkl") -> pd.DataFrame:
    """
    Preprocess a new CSV for inference (no SMOTE, no split).
    Expects same columns as training data.
    """
    df = df.copy()
    scaler = joblib.load(scaler_path)
    df["Amount_scaled"] = scaler.transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)
    if "Class" in df.columns:
        df.drop(columns=["Class"], inplace=True)
    return df
