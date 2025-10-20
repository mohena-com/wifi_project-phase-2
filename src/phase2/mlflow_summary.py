import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("10_EPOCHS_TEST_DIFF_BATCH_SIZE")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Select relevant columns
cols = ["tags.mlflow.runName", "params.model_name", "params.lr", "params.batch_size",
        "params.optimizer", "params.weight_decay", "params.epochs",
        "metrics.train_acc", "metrics.val_acc", "metrics.val_f1"]

summary_df = runs[cols].rename(columns={
    "tags.mlflow.runName": "Run Name",
    "params.model_name": "Model",
    "params.lr": "Learning Rate",
    "params.batch_size": "Batch Size",
    "params.optimizer": "Optimizer",
    "params.weight_decay": "Weight Decay",
    "params.epochs": "Epochs",
    "metrics.train_acc": "Train Acc",
    "metrics.val_acc": "Val Acc",
    "metrics.val_f1": "F1 Score"
})

# Sort by Val Acc descending
summary_df = summary_df.sort_values(by="Val Acc", ascending=False)
print(summary_df)

# Save as CSV or Excel
summary_df.to_csv("mlflow_run_summary.csv", index=False)
