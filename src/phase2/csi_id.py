import os
import re
import csv
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from torch.utils.data import random_split
from config_reader import ConfigReader 
from DL_CSILSTMNet import CSILSTMNet
from DS_WifiCSIDataset import WifiCSIDataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU only")


###########################################
# Step 1: CSV parsing (your version, adapted)
###########################################

cr = ConfigReader("csi_id_config.properties")

from datetime import datetime

# Get the current date and time
now = datetime.now()
# Format and print only the time
current_time = now.strftime("%Y%m%d_%H%M%S")

print("Start : Current Time =", current_time)

base_dir = cr.get("local_data_path")
plot_path = f"{cr.get('output_path')}/{current_time}/plots"
log_path = f"{cr.get('output_path')}/{current_time}/logs"
checkpoint_path = f"{cr.get('output_path')}/{current_time}/checkpoints"
checkpoint_dir = checkpoint_path
os.makedirs(checkpoint_dir, exist_ok=True)

print(" Plot path:", plot_path)
print(" log path:", log_path)
print(" best model path:", checkpoint_dir)

os.makedirs(plot_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

import logging
log_filename = f"{log_path}/run_{current_time}.log"
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Example usage:
logger.info("Execution started.")

# For errors or warnings:
# logger.error("An error occurred")
# logger.warning("This is a warning"
 
# You can still use print for console output if needed, but logger will save all logs to the file

import matplotlib.pyplot as plt

def plot_training_history(history, save_path=None):
    """
    Plots accuracy and loss curves for training and validation.

    Args:
        history (dict): Dictionary with keys ['accuracy', 'loss', 'val_accuracy', 'val_loss'].
                        Each key maps to a list/array of metric values per epoch.
        save_path (str or None): If provided, saves the plot image to this path.

    Returns:
        None
    """
    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()



###########################################
# Step 3: Example usage
###########################################
if __name__ == "__main__":

    import glob
    import os

    # Correct glob pattern to find files with 'C03' in name and .csv extension recursively
    gait_filenme = cr.get("file_name_for_gait")
    filelist = glob.glob(os.path.join(base_dir, '**', gait_filenme), recursive=True)
    logger.fatal(f"Found CSV files: { len(filelist)}")
 
    dataset = WifiCSIDataset(logger, filelist, window_size=128, stride=64)
    logger.fatal(f"A. dataset loaded: length : {dataset.__len__()}  ")
    k = 0
    for d in dataset:
        logger.info(f"B. {k}. Sample shapes: {d['metadata_seq'].shape}, { d['csi_seq'].shape}, {d['label'].shape}")
        logger.debug(f"C. {k}. Sample label: {d['metadata_seq']}")
        k = k + 1 

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    logger.info(f"D. metadata_seq: { batch['metadata_seq'].shape}")  # (B, W, 12)
    logger.info(f"E. csi_seq:{ batch['csi_seq'].shape}")            # (B, W, 99)
    logger.info(f"F. label:{ batch['label'].shape}")                # (B,)

    import torch.optim as optim

    # Instantiate model
    model = CSILSTMNet(
        csi_input_size=batch["csi_seq"].shape[2],
        meta_input_size=batch["metadata_seq"].shape[2],
        window_size=batch["metadata_seq"].shape[1],
        num_classes=31  # adjust as needed
    ).to(device)
    logger.fatal(f"LSTM Model Created")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Split dataset into train and test sets (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_epochs = cr.get_int("epochs")
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_val_acc = 0.0  # initialize best validation accuracy
    best_model = None
    best_epoch = 0
    import time
    for epoch in range(num_epochs):

        start_time = time.time()

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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
        train_accuracies.append(correct / total)

        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                csi_seq = batch["csi_seq"].to(device)
                meta_seq = batch["metadata_seq"].to(device)
                labels = batch["label"].squeeze().to(device)
                outputs = model(csi_seq, meta_seq)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                val_acc = correct / total
                # Save model if improvement
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                    best_epoch = epoch + 1
    
        avg_loss = running_loss / len(test_loader)
        test_losses.append(avg_loss)
        test_accuracies.append(correct / total)

        end_time = time.time()
        epoch_time = end_time - start_time
        logger.fatal(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_accuracies[-1]:.4f} | Time: {epoch_time:.2f} sec")

    # serialize best model
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{best_epoch}.pt")
    torch.save(best_model.state_dict(), checkpoint_path)
    logger.fatal(f"Saved new best model at epoch {best_epoch} to {checkpoint_path}")

    # Prepare the history dictionary
    history = {
        'accuracy': train_accuracies,
        'val_accuracy': test_accuracies,
        'loss': train_losses,
        'val_loss': test_losses
    }

    # Call the plot function and save the plot
    plot_training_history(history, save_path=f"{plot_path}/training_history.png")

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_path}/loss_vs_epoch.png")

    # Plot accuracy curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, num_epochs+1), test_accuracies, label="Test Accuracy", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_path}/accuracy_vs_epoch.png")

    # Evaluate on test set and print stats
    all_preds = []
    all_labels = []
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

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    test_accuracy = np.mean(all_preds == all_labels)
    logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion Matrix:\n { cm}")

    # Classification Report
    logger.info(f"Classification Report:\n { classification_report(all_labels, all_preds)}")

    # Macro Precision, Recall, F1-score
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    logger.info(f"Macro Precision: {precision:.4f}, Macro Recall: {recall:.4f}, Macro F1-score: {f1:.4f}")

    # Per-class Accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    logger.info(f"Per-class Accuracy:{ per_class_acc}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{plot_path}/confusion_matrix.png")
    plt.close()

current_time = now.strftime("%Y%m%d_%H%M%S")

print(f"End : Current Time ={ current_time}")

# https://www.perplexity.ai/search/apply-pca-on-a-dataset-for-dim-XHDCWLULS3S6upRT9nu0ig#31