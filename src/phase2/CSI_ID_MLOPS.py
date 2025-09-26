import os, glob, time, itertools, logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Import your custom classes
from config_reader import ConfigReader
from DS_WifiCSIDataset import WifiCSIDataset
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_MobileNetV3 import MobileNetV3_1D_LSTM
# EfficientNet1D would be imported similarly

def train_and_evaluate(model, train_loader, val_loader, device, params, logger):
    """Train and evaluate for one set of params, return metrics, best ckpt, and full history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    num_epochs = params["epochs"]
    best_val_acc, best_epoch = 0, 0
    best_model_state = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(num_epochs):
        start_time = time.time()
        # Training
        model.train(); running_loss, correct, total = 0.0, 0, 0
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
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_acc = correct / total
        train_accs.append(train_acc)
        # Validation
        model.eval(); running_loss, correct, total = 0.0, 0, 0
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
        avg_val_loss = running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        val_accs.append(val_acc)
        end_time = time.time()
        mlflow.log_metrics({'train_loss': avg_loss, 'train_acc': train_acc,
                            'val_loss': avg_val_loss, 'val_acc': val_acc}, step=epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train: {train_acc:.4f}, Val: {val_acc:.4f} | Time: {end_time-start_time:.2f}s")
        # Best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), f"best_model_{params['model_name']}_epoch{best_epoch}.pt")
            mlflow.pytorch.log_model(model, f"best_model_{params['model_name']}")
    return (train_losses, val_losses, train_accs, val_accs, best_model_state, best_val_acc, best_epoch)

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
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    logger = logging.getLogger()

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
    mlflow.set_experiment("CSI_WiFi_MLOps")
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
                    model, train_loader, test_loader, device, params, logger)
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
                    torch.save(best_model_state, os.path.join(checkpoint_dir, f"best_model_{model_name}_epoch{best_epoch}.pt"))
                mlflow.log_metric("best_val_acc", best_val_acc)
    print(f"Best model overall: {best_overall}")
    logger.info(f"Best model overall: {best_overall}")

if __name__ == "__main__":
    run_mlop_pipeline()
