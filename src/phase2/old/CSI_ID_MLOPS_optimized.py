import os
import glob
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import psutil
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from mlflow.models.signature import infer_signature
from typing import Dict, Any, Tuple, List

# Import your custom classes
from config_reader import ConfigReader
from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM

def setup_logging(log_file: str, log_level: str) :

    l_level = None

    if log_level.upper() == "DEBUG":
        l_level = logging.DEBUG
    elif log_level.upper() == "INFO":
        l_level = logging.INFO
    elif log_level.upper() == "WARNING":
        l_level = logging.WARNING
    elif log_level.upper() == "ERROR":  
        l_level = logging.ERROR
    elif log_level.upper() == "CRITICAL":
        l_level = logging.CRITICAL
    elif log_level.upper() == "FATAL":
        l_level = logging.CRITICAL
    else:
        l_level = logging.INFO

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=l_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
    )
    logger = logging.getLogger(__name__)
    return logger


def setup_environment(config_path: str) -> Dict[str, Any]:
    """
    Sets up the environment by reading config, creating directories, and initializing logging.
    Returns a dictionary of configuration settings and paths.
    """
    cr = ConfigReader(config_path)
    now = time.strftime("%Y%m%d_%H%M%S")
    
    # Consolidate paths
    output_path = cr.get('output_path')
    exp_name = cr.get('experiment_name')
    run_path = os.path.join(output_path, now)
    
    paths = {
        "plots": os.path.join(run_path, "plots"),
        "logs": os.path.join(run_path, "logs"),
        "checkpoints": os.path.join(run_path, "checkpoints"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # Setup Logging
    log_level = cr.get('log_level')
    log_file = os.path.join(paths["logs"], f"{exp_name}_run_{now}.log")
    logger = setup_logging(log_file, log_level)
    logger.info("Environment setup complete.")

    # Consolidate configuration
    config = {
        "cr": cr,
        "paths": paths,
        "logger": logger,
        "device": torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"),
        "filelist": glob.glob(os.path.join(cr.get("local_data_path"), '**', cr.get("file_name_for_gait")), recursive=True)
    }
    
    logger.info(f"Using device: {config['device']}")
    return config

def get_dataloaders(logger: logging.Logger, filelist: List[str], batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepares and returns train and test dataloaders."""
    dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def get_model(model_name: str, sample_batch: Dict[str, torch.Tensor], num_classes: int) -> nn.Module:
    """
    Model factory to instantiate a model based on its name and data shape.
    """
    csi_shape = sample_batch["csi_seq"].shape
    meta_shape = sample_batch["metadata_seq"].shape

    if model_name == "CSILSTMNet":
        return CSILSTMNet(
            csi_input_size=csi_shape[2],
            meta_input_size=meta_shape[2],
            window_size=meta_shape[1],
            num_classes=num_classes
        )
    elif model_name in ["DenseNet1D", "MobileNetV3_1D_LSTM"]:
        model_class = DenseNet1D if model_name == "DenseNet1D" else MobileNetV3_1D_LSTM
        return model_class(
            csi_channels=csi_shape[2],
            meta_feature_dim=meta_shape[2],
            num_classes=num_classes
        )
    elif model_name == "EfficientNet1DLSTM":
        return EfficientNet1DLSTM(
            # NOTE: Corrected parameter names based on typical model design
            in_channels=csi_shape[2],
            meta_feature_dim=meta_shape[2],
            num_classes=num_classes,
            meta_seq_len=meta_shape[1] # Pass this if your model needs it
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_and_evaluate(model: nn.Module, params: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """
    Trains and evaluates a model for one full epoch cycle.
    Logs metrics to MLflow and returns the best results.
    """
    logger.info(f"Starting training for {params['model_name']} with LR={params['lr']}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    
    best_val_acc = 0.0
    best_epoch = -1
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    mlflow.log_param("parameter_count", sum(p.numel() for p in model.parameters()))

    for epoch in range(params["epochs"]):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            csi_seq = batch["csi_seq"].to(device, non_blocking=True)
            meta_seq = batch["metadata_seq"].to(device, non_blocking=True)
            labels = batch["label"].squeeze().to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(csi_seq, meta_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # --- Validation Phase ---
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                csi_seq = batch["csi_seq"].to(device, non_blocking=True)
                meta_seq = batch["metadata_seq"].to(device, non_blocking=True)
                labels = batch["label"].squeeze().to(device, non_blocking=True)
                outputs = model(csi_seq, meta_seq)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # --- Logging ---
        mlflow.log_metrics({
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc
        }, step=epoch)
        
        epoch_duration = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{params['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        # --- Save Best Model State ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}.")
    
    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "history": history
    }

def generate_and_log_artifacts(model, best_model_state, test_loader, device, paths, model_name, params, history, logger):
    """Generates plots, reports, and logs them as MLflow artifacts."""
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # --- Plot training history ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss'); plt.legend(); plt.grid(True)
    stats_path = os.path.join(paths['plots'], f"{model_name}_{params['lr']}_stats.png")
    plt.savefig(stats_path)
    plt.close()
    mlflow.log_artifact(stats_path)

    # --- Evaluate on test set ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            csi_seq = batch["csi_seq"].to(device, non_blocking=True)
            meta_seq = batch["metadata_seq"].to(device, non_blocking=True)
            labels = batch["label"].squeeze().cpu().numpy()
            outputs = model(csi_seq, meta_seq)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # --- Log CM and Classification Report ---
    cm = confusion_matrix(all_labels, all_preds)
    cr_report_str = classification_report(all_labels, all_preds)
    logger.info(f"Classification Report:\n{cr_report_str}")
    
    # Save classification report as a text file artifact
    cr_path = os.path.join(paths['plots'], f"{model_name}_{params['lr']}_report.txt")
    with open(cr_path, 'w') as f:
        f.write(cr_report_str)
    mlflow.log_artifact(cr_path)

    # Plot and save confusion matrix
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    cm_path = os.path.join(paths['plots'], f"{model_name}_{params['lr']}_cm.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

def run_experiment():
    """Main function to orchestrate the MLOps pipeline."""
    env_config = setup_environment("csi_id_config.properties")
    logger = env_config["logger"]
    
    train_loader, test_loader = get_dataloaders(logger, env_config["filelist"], batch_size=16)

    # --- Model & Hyperparameter Definitions ---
    model_defs = {
        "CSILSTMNet": {'lr': [0.001, 0.0005]},
        "DenseNet1D": {'lr': [0.001, 0.0005]},
        "EfficientNet1DLSTM": {'lr': [0.001, 0.0005]},
        "MobileNetV3_1D_LSTM": {'lr': [0.001, 0.0005]},
    }
    
    mlflow.set_experiment(env_config["cr"].get("experiment_name"))
    
    sample_batch = next(iter(train_loader))
    num_classes = 31 # Assuming this is fixed, or derive from dataset

    for model_name, param_grid in model_defs.items():
        # --- Parent Run for each model type ---
        with mlflow.start_run(run_name=f"{model_name}_main") as parent_run:
            logger.info(f"--- Starting experiments for {model_name} ---")
            mlflow.log_param("model_architecture", model_name)
            
            param_keys, param_vals = zip(*param_grid.items())
            
            for combo in itertools.product(*param_vals):
                params = dict(zip(param_keys, combo))
                params['model_name'] = model_name
                params['epochs'] = env_config["cr"].get_int("epochs")
                
                # --- Child Run for each hyperparameter combination ---
                with mlflow.start_run(run_name=f"lr_{params['lr']}", nested=True) as child_run:
                    logger.info(f"Starting trial with params: {params}")
                    mlflow.log_params({k: v for k, v in params.items() if k != 'model_name'})
                    
                    # --- Model Initialization and Training ---
                    model = get_model(model_name, sample_batch, num_classes).to(env_config["device"])
                    
                    results = train_and_evaluate(model, params, train_loader, test_loader, env_config["device"], logger)

                    # --- Log Final Metrics and Artifacts ---
                    if results["best_model_state"]:
                        mlflow.log_metric("best_val_accuracy", results["best_val_acc"])
                        mlflow.log_metric("best_epoch", results["best_epoch"])
                        
                        generate_and_log_artifacts(
                            model, results["best_model_state"], test_loader, 
                            env_config["device"], env_config["paths"], model_name, 
                            params, results["history"], logger
                        )
                        
                        # --- Log the final model with signature ---
                        signature = infer_signature(
                            sample_batch["csi_seq"].numpy(), 
                            model(sample_batch["csi_seq"].to(env_config["device"]), 
                            sample_batch["metadata_seq"].to(env_config["device"])).detach().cpu().numpy()
                        )
                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            artifact_path=f"best_{model_name}",
                            signature=signature
                        )
                    else:
                        logger.warning("No best model was saved. Skipping artifact logging.")

if __name__ == "__main__":
    run_experiment()