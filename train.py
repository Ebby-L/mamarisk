"""Train MamaRisk: compare models with cross-validation, then fit and save the best.

Usage:
    python train.py                 # downloads the UCI dataset on first run
    python train.py --data my.csv   # use a local copy instead
"""
import argparse

from mamarisk.data import load_clean, load_raw
from mamarisk.model import evaluate, save, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to a local CSV of the dataset")
    args = ap.parse_args()

    raw = load_raw(args.data)
    df = load_clean(args.data)
    print(f"Loaded {len(raw)} rows; {len(df)} after removing exact duplicates.")
    print(df["RiskLevel"].value_counts().to_string(), "\n")

    results = evaluate(df)
    print("5-fold cross-validation on de-duplicated data:")
    print(results.round(3).to_string(index=False), "\n")

    best = results.iloc[0]
    pipe = train(df, best["model"])
    path = save(pipe, best["model"], best.drop("model").round(4).to_dict())
    print(f"Saved {best['model']} to {path}")


if __name__ == "__main__":
    main()
