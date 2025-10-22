import mlflow
import pandas as pd

# Connect to MLflow and fetch runs
client = mlflow.tracking.MlflowClient()
exp_name = "10_EPOCHS_TEST_DIFF_BATCH_SIZE"
experiment = client.get_experiment_by_name(exp_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print("🔍 Available MLflow columns:\n")
print(runs.columns.tolist())

print("\n🔍 Sample of run data:\n")
print(runs.head())

# ✅ Select only columns that actually exist in your MLflow run data
cols = [
    'run_id', 'experiment_id', 'status', 'artifact_uri', 'start_time', 'end_time',
    'metrics.Training Accuracy Curve', 'metrics.Validation Accuracy Curve',
    'metrics.Training F1 Score', 'metrics.Validation F1 Score',
    'metrics.Training Recall Curve', 'metrics.Validation Recall Curve',
    'metrics.Training Precision Curve', 'metrics.Validation Precision Curve',
    'metrics.Top Validation Accuracy',
    'params.model_name', 'params.lr', 'params.batch_size', 'params.optimizer',
    'params.weight_decay', 'params.epochs', 'params.window_size'
]

# ✅ Filter the DataFrame to include only available columns
cols = [c for c in cols if c in runs.columns]
summary_df = runs[cols].copy()

# ✅ Rename for clarity
summary_df = summary_df.rename(columns={
    "params.model_name": "Model",
    "params.lr": "Learning Rate",
    "params.batch_size": "Batch Size",
    "params.optimizer": "Optimizer",
    "params.weight_decay": "Weight Decay",
    "params.epochs": "Epochs",
    "metrics.Training Accuracy Curve": "Train Accuracy",
    "metrics.Validation Accuracy Curve": "Val Accuracy",
    "metrics.Training F1 Score": "Train F1",
    "metrics.Validation F1 Score": "Val F1",
    "metrics.Top Validation Accuracy": "Best Val Accuracy"
})

# ✅ Sort by Validation Accuracy if available
if "Val Accuracy" in summary_df.columns:
    summary_df = summary_df.sort_values(by="Val Accuracy", ascending=False)

print("\n📊 Experiment Summary Table:\n")
print(summary_df)

# ✅ Save as CSV
summary_df.to_csv(f"../../docs/{exp_name}_mlflow_run_summary.csv", index=False)
print(f"\n✅ Saved summary table to ../../docs/{exp_name}_mlflow_run_summary.csv")
