"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic credit card fraud dataset
when the real Kaggle dataset is not available.

Usage:
    python data/generate_sample_data.py
    python data/generate_sample_data.py --rows 50000 --fraud_rate 0.002
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def generate_creditcard_like(n_rows: int = 100_000,
                              fraud_rate: float = 0.00172,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mimics the ULB Credit Card Fraud dataset.

    Structure:
      - Time   : seconds elapsed from first transaction (0 to ~172800)
      - V1–V28 : PCA-like features with fraud-specific signal injected
      - Amount : transaction amount in euros
      - Class  : 0=normal, 1=fraud
    """
    rng = np.random.default_rng(random_state)
    n_fraud  = max(1, int(n_rows * fraud_rate))
    n_normal = n_rows - n_fraud

    print(f"Generating synthetic dataset:")
    print(f"  Total rows  : {n_rows:,}")
    print(f"  Normal rows : {n_normal:,}")
    print(f"  Fraud rows  : {n_fraud:,} ({fraud_rate*100:.3f}%)")

    # ── normal transactions ──────────────────────────────────────────────────
    V_normal = rng.standard_normal((n_normal, 28))
    # Inject mild correlations (mimic PCA structure)
    for i in range(27):
        V_normal[:, i+1] += 0.1 * V_normal[:, i]

    amount_normal = np.abs(rng.lognormal(mean=4.2, sigma=1.8, size=n_normal))
    amount_normal = np.clip(amount_normal, 0.01, 25_000)

    # ── fraud transactions ───────────────────────────────────────────────────
    V_fraud = rng.standard_normal((n_fraud, 28))
    # Known fraud signals: V14, V12, V10, V4, V11 are highly discriminating
    fraud_signal = {
        3:  -4.5,   # V4  (high negative = fraud signal)
        9:  -5.0,   # V10
        10: -3.2,   # V11
        11: -6.5,   # V12
        13: -8.0,   # V14
        16: +3.8,   # V17
        17: -3.5,   # V18
    }
    for col, shift in fraud_signal.items():
        V_fraud[:, col] += shift + rng.standard_normal(n_fraud) * 0.6

    # Fraud amounts: typically small (testing) or unusually large
    amount_fraud = np.where(
        rng.random(n_fraud) < 0.65,
        rng.uniform(1, 200, n_fraud),          # small test amounts
        np.abs(rng.lognormal(5.5, 1.0, n_fraud))  # large amounts
    )
    amount_fraud = np.clip(amount_fraud, 0.01, 10_000)

    # ── combine ──────────────────────────────────────────────────────────────
    V_all      = np.vstack([V_normal, V_fraud])
    amounts    = np.concatenate([amount_normal, amount_fraud])
    labels     = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)

    # Simulate Time (two transaction density peaks per day)
    time_all = np.sort(rng.uniform(0, 172800, n_rows))

    col_names = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(V_all, columns=col_names)
    df.insert(0, "Time",   time_all)
    df["Amount"] = amounts
    df["Class"]  = labels

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--rows",       type=int,   default=100_000, help="Number of rows")
    parser.add_argument("--fraud_rate", type=float, default=0.00172, help="Fraud proportion")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out",        type=str,   default="data/creditcard.csv")
    args = parser.parse_args()

    df = generate_creditcard_like(args.rows, args.fraud_rate, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print("\nFirst 5 rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
