import os
import glob
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import psutil
from mlflow.models.signature import infer_signature
import itertools
from sklearn.preprocessing import label_binarize

# Import your custom classes
from config_reader import ConfigReader
from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM
# EfficientNet1D would be imported similarly

# --- Logging setup (use your CSI_ID.py pattern) ---
cr = ConfigReader("csi_id_config.properties")
now = time.strftime("%Y%m%d_%H%M%S")
base_dir = cr.get("local_data_path")
gait_filenme = cr.get("file_name_for_gait")
filelist = glob.glob(os.path.join(base_dir, '**', gait_filenme), recursive=True)
plot_path = f"{cr.get('output_path')}/{now}/plots"; os.makedirs(plot_path, exist_ok=True)
print(f"Plot path: {plot_path}")
log_path = f"{cr.get('output_path')}/{now}/logs"; os.makedirs(log_path, exist_ok=True)
print(f"Log path: {log_path}")
checkpoint_dir = f"{cr.get('output_path')}/{now}/checkpoints"; os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoint path: {checkpoint_dir}") 
log_filename = f"{log_path}/run_{now}.log"
print(f"Log file: {log_filename}")



def setup_logging(log_file_path):

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, mode="w"),
        ],
    )
    logger = logging.getLogger()
    logger.info("Logger initialized")
    return logger

def train_and_evaluate(model, train_loader, val_loader, device, params, checkpoint_dir, logger):
    """Train and evaluate for one set of params, return metrics, best ckpt, and full history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    num_epochs = params["epochs"]
    best_val_acc, best_epoch = 0, 0
    best_model_state = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    total_params = sum(p.numel() for p in model.parameters())
    mlflow.log_param("parameter_count", total_params)
    for epoch in range(num_epochs):
        start_time = time.time()
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_true, train_pred = [], []
        for batch in train_loader:
            csi_seq = batch["csi_seq"].to(device)
            meta_seq = batch["metadata_seq"].to(device)
            labels = batch["label"].squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(csi_seq, meta_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_true.extend(labels.cpu().numpy())
            train_pred.extend(preds.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        train_precision = precision_score(train_true, train_pred, average="weighted", zero_division=0)
        train_recall = recall_score(train_true, train_pred, average="weighted", zero_division=0)
        train_f1 = f1_score(train_true, train_pred, average="weighted", zero_division=0)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_precision", train_precision, step=epoch)
        mlflow.log_metric("train_recall", train_recall, step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)

        # Validation phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        val_true, val_pred, val_prob = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                csi_seq = batch["csi_seq"].to(device)
                meta_seq = batch["metadata_seq"].to(device)
                labels = batch["label"].squeeze().to(device)
                outputs = model(csi_seq, meta_seq)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                val_true.extend(labels.cpu().numpy())
                val_pred.extend(preds.cpu().numpy())
                val_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        val_precision = precision_score(val_true, val_pred, average="weighted", zero_division=0)
        val_recall = recall_score(val_true, val_pred, average="weighted", zero_division=0)
        val_f1 = f1_score(val_true, val_pred, average="weighted", zero_division=0)

        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        val_true_np = np.array(val_true)
        val_prob_np = np.array(val_prob)

        unique_classes = np.unique(val_true_np)
        num_classes = val_prob_np.shape[1]

        if len(unique_classes) == num_classes:
        # All classes present, compute directly
            try:
                val_auc = roc_auc_score(val_true_np, val_prob_np, multi_class='ovr', average='weighted')
                mlflow.log_metric('val_auc', val_auc, step=epoch)
            except Exception as e:
                logger.warning(f"ROC-AUC computation failed: {e}")
        else:
            #    Subset classes and probabilities to avoid mismatch error
            try:
                y_true_bin = label_binarize(val_true_np, classes=unique_classes)
                val_prob_subset = val_prob_np[:, unique_classes]
                val_auc = roc_auc_score(y_true_bin, val_prob_subset, multi_class='ovr', average='weighted')
                mlflow.log_metric('val_auc', val_auc, step=epoch)
            except Exception as e:
                logger.warning(f"ROC-AUC subset computation failed: {e}")

        for param_group in optimizer.param_groups:
            mlflow.log_metric("learning_rate", param_group["lr"], step=epoch)

        end_time = time.time()
        mlflow.log_metric("epoch_time_sec", end_time - start_time, step=epoch)

        mlflow.log_metric("cpu_percent", psutil.cpu_percent())
        mlflow.log_metric("memory_used_gb", psutil.virtual_memory().used / (1024 ** 3))
        if torch.cuda.is_available():
            mlflow.log_metric("gpu_mem_allocated_gb", torch.cuda.memory_allocated() / (1024 ** 3))
            mlflow.log_metric("gpu_mem_reserved_gb", torch.cuda.memory_reserved() / (1024 ** 3))

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Epoch Duration: {end_time-start_time:.2f} sec"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            mlflow.pytorch.log_model(model, f"best_model_{params['model_name']}")
			
       # Log model with signature and input example for MLflow
    '''    
    for p in params:
        print(f"Params : {p} : {params[p]} ")
    
    dummy_csi = torch.randn(1, 128, params['csi_input_size']).to(device)
    dummy_meta = torch.randn(1, params['meta_input_size']).to(device)
    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_csi, dummy_meta)
    signature = infer_signature(
        inputs={"csi_seq": dummy_csi.cpu().numpy(), "meta_seq": dummy_meta.cpu().numpy()},
        outputs=dummy_output.cpu().numpy()
    )
    mlflow.pytorch.log_model(
        model,
        artifact_path=f"best_model_{params['model_name']}",
        input_example={"csi_seq": dummy_csi.cpu().numpy(), "meta_seq": dummy_meta.cpu().numpy()},
        signature=signature
    )
	'''	
    logger.fatal(f"Training complete. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}.")
    if best_model_state is not None:
        torch.save(best_model_state, f"{checkpoint_dir}/best_model_{params['model_name']}_epoch{best_epoch}.pt")
        logger.fatal(f"00. Saved Best Model. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}.")
        mlflow.log_artifact(f"{checkpoint_dir}/best_model_{params['model_name']}_epoch{best_epoch}.pt")

    return train_losses, val_losses, train_accs, val_accs, best_model_state, best_val_acc, best_epoch

        #logger.info(f"Best Val Acc so far: {best_val_acc:.4f} at epoch {best_epoch} best_model_state : {best_model_state}")

def plot_stats(history, save_path, logger):
    epochs = range(1, len(history['accuracy']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    mlflow.log_artifact(save_path)
    logger.info(f"Saved stats plot to {save_path}")
    plt.close()




def run_mlop_pipeline():
    logger = setup_logging(log_filename)
    print(f"logger {logger}")
    # --- Dataset loading (as in DS_WifiCSIDataset.py) ---
    dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         "cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")  
    # --- Model Definitions & hyperparameter grid ---
    model_defs = {
        "CSILSTMNet": (CSILSTMNet, {'csi_input_size':[99], 'meta_input_size':[12], 'window_size':[128], 'num_classes':[31], 'lr':[0.001, 0.0005], 'epochs':[cr.get_int("epochs")]}),
        "DenseNet1D": (DenseNet1D, {'csi_channels':[99], 'meta_feature_dim':[12], 'num_classes':[31], 'lr':[0.001, 0.0005], 'epochs':[cr.get_int("epochs")]}),
        "EfficientNet1DLSTM": (EfficientNet1DLSTM, {'in_channels':[99], 'meta_seq_len':[128], 'meta_feature_dim':[12], 'num_classes':[31], 'lr':[0.001, 0.0005], 'epochs':[cr.get_int("epochs")]}),
        "MobileNetV3_1D_LSTM": (MobileNetV3_1D_LSTM, {'csi_channels':[99], 'meta_feature_dim':[12], 'num_classes':[31], 'lr':[0.001, 0.0005], 'epochs':[cr.get_int("epochs")]}),
        # If EfficientNet1D is available, add here
    }

    # --- MLflow experiment ---
    mlflow.set_experiment(cr.get("experiment_name"))
    best_overall = {"val_acc":-1}
    stats_summary = {}

    for model_name, (model_class, param_grid) in model_defs.items():
        param_keys, param_vals = zip(*param_grid.items())
        for combo in itertools.product(*param_vals):
            params = dict(zip(param_keys, combo)); params['model_name'] = model_name
            with mlflow.start_run(run_name=f"{model_name}_{str(params)}"):
                batch = next(iter(train_loader))
                # Model instantiation according to constructor
                if model_name == "CSILSTMNet":
                    model = model_class(
                        csi_input_size=batch["csi_seq"].shape[2],
                        meta_input_size=batch["metadata_seq"].shape[2],
                        window_size=batch["metadata_seq"].shape[1],
                        num_classes=31
                    ).to(device)
                elif model_name in ["DenseNet1D", "MobileNetV3_1D_LSTM"]:
                    model = model_class(
                        csi_channels=batch["csi_seq"].shape[2],
                        meta_feature_dim=batch["metadata_seq"].shape[2],
                        num_classes=31
                    ).to(device)
                elif model_name == "EfficientNet1DLSTM":
                    model = model_class(
                        csi_input_channels=batch["csi_seq"].shape[2],
                        meta_input_size=batch["metadata_seq"].shape[-1],
                        num_classes=31
                    ).to(device)
                else:
                    raise ValueError("Unknown model")
                mlflow.log_params(params)
                train_losses, val_losses, train_accs, val_accs, best_model_state, best_val_acc, best_epoch = train_and_evaluate(
                    model, train_loader, test_loader, device, params, checkpoint_dir, logger)
                #logger.info(f"best model {best_model_state}")
                history = {
                    'accuracy': train_accs,
                    'val_accuracy': val_accs,
                    'loss': train_losses,
                    'val_loss': val_losses
                }
                stats_summary[(model_name, str(params))] = {
                    "val_acc": best_val_acc,
                    "epoch": best_epoch,
                    "history": history
                }
                plot_stats(history, save_path=f"{plot_path}/{model_name}_{str(params)}_stats.png", logger=logger)
                # Save confusion matrix, classification report on test set
                if best_model_state is None:
                    logger.warning("No best model state found, skipping test evaluation.")
                    continue
                model.load_state_dict(best_model_state)
                all_preds, all_labels = [], []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        csi_seq = batch["csi_seq"].to(device)
                        meta_seq = batch["metadata_seq"].to(device)
                        labels = batch["label"].squeeze().cpu().numpy()
                        outputs = model(csi_seq, meta_seq)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(labels)
                cm = confusion_matrix(all_labels, all_preds); cr_report = classification_report(all_labels, all_preds)
                np.save(f"{plot_path}/{model_name}_{str(params)}_cm.npy", cm)
                mlflow.log_artifact(f"{plot_path}/{model_name}_{str(params)}_cm.npy")
                logger.info(f"Confusion matrix:\n{cm}")
                logger.info(f"Classification report:\n{cr_report}")
                # Save model checkpoint for best overall if needed
                if best_val_acc > best_overall["val_acc"]:
                    best_overall = {"model": model_name, "params": params, "val_acc": best_val_acc, "epoch": best_epoch}
                    torch.save(best_model_state, f"{checkpoint_dir}/best_model_{params['model_name']}_epoch{best_epoch}.pt")
                    logger.fatal(f"99. Saved Best Model. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}.")
                    mlflow.log_artifact(f"{checkpoint_dir}/best_model_{params['model_name']}_epoch{best_epoch}.pt")

                mlflow.log_metric("best_val_acc", best_val_acc)
  #  print(f"Best model overall: {best_overall}")
    logger.info(f"Best model overall: {best_overall}")

if __name__ == "__main__":
    run_mlop_pipeline()


# for additional metrices - 
# https://www.perplexity.ai/search/apply-pca-on-a-dataset-for-dim-XHDCWLULS3S6upRT9nu0ig#54