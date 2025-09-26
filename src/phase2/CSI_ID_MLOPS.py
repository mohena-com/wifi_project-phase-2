import os
import itertools
import time
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
import psutil
import joblib

# Import all your model class files here
from DL_EfficientNet1DLSTM import EfficientNet1DLSTM
from DL_CSILSTMNet import CSILSTMNet
from DL_DenseNet1D import DenseNet1D
from DL_MobileNetV3 import MobileNetV3_1D_LSTM

# Add EfficientNet1D, if separate, import accordingly

from torch.utils.data import DataLoader

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for batch in dataloader:
        csi_seq = batch['csi_seq'].to(device)
        meta_seq = batch['metadata_seq'].to(device)
        labels = batch['label'].squeeze().to(device)
        optimizer.zero_grad()
        outputs = model(csi_seq, meta_seq)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            csi_seq = batch['csi_seq'].to(device)
            meta_seq = batch['metadata_seq'].to(device)
            labels = batch['label'].squeeze().to(device)
            outputs = model(csi_seq, meta_seq)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return running_loss / total, correct / total

def log_system_metrics():
    mlflow.log_metric("cpu_usage_percent", psutil.cpu_percent(interval=1))
    mem = psutil.virtual_memory()
    mlflow.log_metric("memory_usage_gb", mem.used / (1024**3))
    disk = psutil.disk_usage('/')
    mlflow.log_metric("disk_usage_gb", disk.used / (1024**3))

def grid_search_and_mlflow(
    model_defs, # dict: name: (class, param_grid_dict)
    train_loader, val_loader, device, 
    result_path='./results'
):
    os.makedirs(result_path, exist_ok=True)
    mlflow.set_experiment('CSI_WiFi_Deep_Model_Tuning')
    best_global = {'val_acc': 0}
    all_results = []

    for model_name, (model_class, param_grid) in model_defs.items():
        best_val_acc = 0.0
        best_model_state = None
        print(f"\n[INFO] Running Model: {model_name}")

        # Grid search for this model
        keys, values = zip(*param_grid.items())
        for run_idx, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            run_name = f"{model_name}_run{run_idx+1}"
            with mlflow.start_run(run_name=run_name):
                # ---- Instantiate ----
                model = model_class(**{k: v for k, v in params.items() if not k.startswith('lr') and not k.startswith('epochs')}).to(device)
                optimizer = optim.Adam(model.parameters(), lr=params['lr'])
                criterion = nn.CrossEntropyLoss()
                train_loss_list, val_loss_list = [], []
                train_acc_list, val_acc_list = [], []

                # ---- Train/Evaluate Loop ----
                for epoch in range(params['epochs']):
                    t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                    v_loss, v_acc = evaluate(model, val_loader, criterion, device)
                    train_loss_list.append(t_loss)
                    train_acc_list.append(t_acc)
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                    mlflow.log_metrics({'train_loss': t_loss, 'train_acc': t_acc,
                                       'val_loss': v_loss, 'val_acc': v_acc}, step=epoch)

                    # Save best within this run
                    if v_acc > best_val_acc:
                        best_val_acc = v_acc
                        best_model_state = model.state_dict()
                        torch.save(model.state_dict(), os.path.join(result_path, f"{model_name}_best.pth"))
                        mlflow.pytorch.log_model(model, "best_model")

                # ---- System resource logging, params, plots ----
                mlflow.log_params(params)
                log_system_metrics()
                # Save training curves
                np.save(os.path.join(result_path, f"{model_name}_loss.npy"), np.array([train_loss_list, val_loss_list]))
                np.save(os.path.join(result_path, f"{model_name}_acc.npy"), np.array([train_acc_list, val_acc_list]))

                # ---- MLflow Artifacts ----
                mlflow.log_artifact(os.path.join(result_path, f"{model_name}_loss.npy"))
                mlflow.log_artifact(os.path.join(result_path, f"{model_name}_acc.npy"))

                all_results.append({'model': model_name, 'params': params, 'val_acc': best_val_acc})

                print(f"    [INFO] Finished Run {run_idx+1} {model_name} | Best val_acc: {best_val_acc:.4f}")

                # Track globally best model
                if best_val_acc > best_global['val_acc']:
                    best_global = {'model': model_name, 'params': params, 'val_acc': best_val_acc}
                    joblib.dump(best_model_state, os.path.join(result_path, f"BEST_MODEL_{model_name}.pth"))

    # Print/Return summary
    print("\n------------- Best Overall Model ---------------")
    print(best_global)
    return all_results, best_global

if __name__ == '__main__':
    # ---------------------
    # SETUP DATALOADERS HERE (Use your dataset loader function/classes, e.g., CSI_ID.py)
    from your_dataset_class import WifiCSIDataset  # CHANGE THIS LINE
    train_dataset = WifiCSIDataset(...)           # FILL INIT ARGS
    val_dataset = WifiCSIDataset(...)             # FILL INIT ARGS
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # ---------------------

    # ---- Define Model Search Space ----
    model_defs = {
        "EfficientNet1DLSTM": (EfficientNet1DLSTM, {
            'csi_channels': [99], 'meta_seq_len':[128], 'meta_feature_dim':[12], 'num_classes':[31],
            'lr': [0.001, 0.0005], 'epochs': [20]
        }),
        "CSILSTMNet": (CSILSTMNet, {
            'csi_input_size':[99], 'meta_input_size':[12], 'window_size':[128], 'num_classes':[31],
            'lr':[0.001, 0.0005], 'epochs': [20]
        }),
        "DenseNet1D": (DenseNet1D, {
            'csi_channels':[99], 'meta_feature_dim':[12], 'num_classes':[31],
            'lr':[0.001, 0.0005], 'epochs': [20]
        }),
        "MobileNetV3_1D_LSTM": (MobileNetV3_1D_LSTM, {
            'csi_channels':[99], 'meta_feature_dim':[12], 'num_classes':[31],
            'lr':[0.001, 0.0005], 'epochs':[20]
        }),
        # If you have EfficientNet1D (non-LSTM), add here
    }

    all_results, best_global = grid_search_and_mlflow(model_defs, train_loader, val_loader, device, result_path='./mlflow_results')

    print("[INFO] Detailed model search complete. Best overall result:")
    print(best_global)
