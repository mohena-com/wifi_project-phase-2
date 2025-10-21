"""
prune_low_performance_combos.py

Usage:
    python prune_low_performance_combos.py --csv mlflow_run_summary.csv
"""

import pandas as pd
from pathlib import Path
import argparse

def prune_bad_combinations(df, val_quantile=0.2, overfit_thresh=0.05, std_thresh=0.05):
    df = df.dropna(subset=["Val F1", "Train F1", "Model",
                           "Learning Rate", "Batch Size",
                           "Optimizer", "Weight Decay"])

    group_cols = ["Model", "Learning Rate", "Batch Size", "Optimizer", "Weight Decay"]
    agg = df.groupby(group_cols).agg({
        "Val F1": ["mean", "std", "max"],
        "Train F1": "mean"
    }).reset_index()

    agg.columns = ["Model", "Learning Rate", "Batch Size", "Optimizer", "Weight Decay",
                   "Val_F1_mean", "Val_F1_std", "Val_F1_max", "Train_F1_mean"]

    val_threshold = agg["Val_F1_mean"].quantile(val_quantile)

    bad_mask = (
        (agg["Val_F1_mean"] < val_threshold) |
        ((agg["Train_F1_mean"] - agg["Val_F1_mean"]) > overfit_thresh) |
        (agg["Val_F1_std"] > std_thresh)
    )

    good_df = agg[~bad_mask].copy().sort_values(by="Val_F1_mean", ascending=False)

    # ✅ remove combos with zero / meaningless metrics
    good_df = good_df[
        (good_df["Val_F1_mean"] > 0.05) &
        (good_df["Val_F1_max"] > 0.05) &
        (good_df["Train_F1_mean"] > 0.05)
    ]

    bad_df = agg[bad_mask].copy().sort_values(by="Val_F1_mean", ascending=True)
    return good_df, bad_df



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to MLflow run summary CSV")
    ap.add_argument("--output", type=str, default="promising_combos.csv",
                    help="File to save filtered combinations")
    args = ap.parse_args()

    prune_bad_combinations(args.csv, args.output)
