"""
prune_low_performance_combos.py

Usage:
    python prune_low_performance_combos.py --csv mlflow_run_summary.csv
"""

import pandas as pd
from pathlib import Path
import argparse

def prune_bad_combinations(csv_path, output_path="promising_combos.csv",
                           val_quantile=0.2, overfit_thresh=0.05, std_thresh=0.05):
    """Filter out unstable or low-performing hyperparameter combinations."""

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Val F1", "Train F1", "Model",
                           "Learning Rate", "Batch Size",
                           "Optimizer", "Weight Decay"])

    # Group by unique hyperparameter settings
    group_cols = ["Model", "Learning Rate", "Batch Size", "Optimizer", "Weight Decay"]
    agg = df.groupby(group_cols).agg({
        "Val F1": ["mean", "std", "max"],
        "Train F1": "mean"
    }).reset_index()

    agg.columns = ["Model", "Learning Rate", "Batch Size", "Optimizer", "Weight Decay",
                   "Val_F1_mean", "Val_F1_std", "Val_F1_max", "Train_F1_mean"]

    # Thresholds
    val_threshold = agg["Val_F1_mean"].quantile(val_quantile)

    # Mask: mark as bad
    bad_mask = (
        (agg["Val_F1_mean"] < val_threshold) |
        ((agg["Train_F1_mean"] - agg["Val_F1_mean"]) > overfit_thresh) |
        (agg["Val_F1_std"] > std_thresh)
    )
    good_df = agg[~bad_mask].copy()
    bad_df = agg[bad_mask].copy()

    # Sort by validation mean F1
    good_df = good_df.sort_values(by="Val_F1_mean", ascending=False).reset_index(drop=True)
    bad_df = bad_df.sort_values(by="Val_F1_mean", ascending=True).reset_index(drop=True)

    # Save both lists
    good_df.to_csv(output_path, index=False)
    bad_df.to_csv(Path(output_path).with_name("discarded_combos.csv"), index=False)

    print(f"✅ Saved {len(good_df)} promising combos to {output_path}")
    print(f"🚫 Saved {len(bad_df)} discarded combos to discarded_combos.csv")
    print(f"Val F1 quantile threshold used: {val_threshold:.4f}")
    return good_df, bad_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to MLflow run summary CSV")
    ap.add_argument("--output", type=str, default="promising_combos.csv",
                    help="File to save filtered combinations")
    args = ap.parse_args()

    prune_bad_combinations(args.csv, args.output)
