import os
import re
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

###########################################
# Step 1: CSV parsing (your version, adapted)
###########################################
def parse_complex(s):
    if s is None or s.strip() == "":
        return 0.0 + 0.0j  # treat missing values as 0
    try:
        s = s.replace('+-', '-').replace('-+', '-').replace('i', 'j')
        return complex(s)
    except Exception:
        # If still not parsable, default to 0
        return 0.0 + 0.0j

def extract_S_A_numbers(filename):
    """
    Extract subject (Sxx) and activity (Axx) numbers from filename.
    Example: 'E1_S01_C03_A03_T01.csv' -> (1, 3)
    """
    match = re.search(r'S(\d+).*A(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_csv_as_numpy(filename):
    print("B_00. Loading:", filename)
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames

        csi_cols = [c for c in cols if c.startswith('csi_')]
        meta_cols = [
            'timestamp_low','bfee_count','Nrx','Ntx',
            'rssi_a','rssi_b','rssi_c','agc',
            'perm_1','perm_2','perm_3'
        ]
        
        X_meta, X_csi = [], []
        subj, act = [], []
        s, a = extract_S_A_numbers(os.path.basename(filename))
        for row in reader:
            # metadata
            meta_row = [float(row[c]) for c in meta_cols]
            # CSI as magnitudes
            csi_row = [abs(parse_complex(row[c])) for c in csi_cols]
            X_meta.append(meta_row)
            X_csi.append(csi_row)
            subj.append(s)
            act.append(a)
        
        X_meta = np.array(X_meta, dtype=np.float32)  # (T, 12)
        X_csi = np.array(X_csi, dtype=np.float32)    # (T, 99)
        print(f"B_01. X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
        
        print(f"B_02. Subject: {len(subj)}, Activity: {len(act)}")
        y = {"subject": subj, "activity": act}
        return X_meta, X_csi, y, meta_cols, csi_cols

###########################################
# Step 2: Dataset with sliding windows
###########################################
class WifiCSIDataset(Dataset):
    def __init__(self, file_list, window_size=128, stride=64, scaler_meta=None, scaler_csi=None):
        self.samples = []
        self.window_size = window_size
        self.stride = stride

        # Fit scalers if not provided
        if scaler_meta is None: 
            self.scaler_meta = StandardScaler()
        else: 
            self.scaler_meta = scaler_meta
        if scaler_csi is None: 
            self.scaler_csi = StandardScaler()
        else: 
            self.scaler_csi = scaler_csi

        # First pass: collect all data for scaling
        all_meta, all_csi = [], []
        i = 0
        for f in file_list:
            X_meta, X_csi, _, _, _ = load_csv_as_numpy(f)
            print(f"A_00_{i}. X_meta: {X_meta.shape} X_csi: {X_csi.shape}")
            all_meta.append(X_meta)
            all_csi.append(X_csi)
            i = i+1
        
        print(f"A_01. all_meta: {(len(all_meta[0]))} all_csi: {len(all_csi)}")

        all_meta = np.vstack(all_meta)
        all_csi = np.vstack(all_csi)
        self.scaler_meta.fit(all_meta)
        self.scaler_csi.fit(all_csi)
        print(f"A_02. all_meta: {all_meta.shape} all_csi: {all_csi.shape}")

        # Second pass: windowed sequences
        for f in file_list:
            X_meta, X_csi, y, _, _ = load_csv_as_numpy(f)
            X_meta = self.scaler_meta.transform(X_meta)
            X_csi = self.scaler_csi.transform(X_csi)

            T = len(X_meta)
            for start in range(0, T - window_size + 1, stride):
                m_seq = X_meta[start:start+window_size]   # (W, 12)
                csi_seq = X_csi[start:start+window_size]  # (W, 99)
                self.samples.append((m_seq, csi_seq, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        m_seq, csi_seq, y = self.samples[idx]
        # Use the first subject label in the window (or last, depending on your task)
        subject_label = y["subject"][0]  # or y["subject"][-1]
        return {
            "metadata_seq": torch.tensor(m_seq, dtype=torch.float32), # (W, 12)
            "csi_seq": torch.tensor(csi_seq, dtype=torch.float32),   # (W, 99)
            "label": torch.tensor(subject_label, dtype=torch.long)   # shape: ()
        }
import torch.nn as nn

class CSILSTMNet(nn.Module):
    def __init__(self, csi_input_size, meta_input_size, window_size, num_classes, hidden_size=64, num_layers=2):
        super().__init__()
        self.csi_lstm = nn.LSTM(csi_input_size, hidden_size, num_layers, batch_first=True)
        self.meta_lstm = nn.LSTM(meta_input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, csi_seq, meta_seq):
        # csi_seq: (B, W, csi_input_size)
        # meta_seq: (B, W, meta_input_size)
        _, (h_csi, _) = self.csi_lstm(csi_seq)   # h_csi: (num_layers, B, hidden_size)
        _, (h_meta, _) = self.meta_lstm(meta_seq)
        # Use last layer's hidden state
        h_csi_last = h_csi[-1]   # (B, hidden_size)
        h_meta_last = h_meta[-1] # (B, hidden_size)
        combined = torch.cat([h_csi_last, h_meta_last], dim=1)  # (B, hidden_size*2)
        out = self.fc(combined)  # (B, num_classes)
        return out

import matplotlib.pyplot as plt

###########################################
# Step 3: Example usage
###########################################
if __name__ == "__main__":
     
    base_dir = "/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/temp"
    #    Usage:
    #base_dir = "/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset"

    import glob
    import os

    # Correct glob pattern to find files with 'C03' in name and .csv extension recursively
    filelist = glob.glob(os.path.join(base_dir, '**', '*C03*.csv'), recursive=True)

    print("Found CSV files:", len(filelist))
 
    dataset = WifiCSIDataset(filelist, window_size=128, stride=64)
    print(f"A. dataset: length : {dataset.__len__()}  ")
    k = 0
    for d in dataset:
        print(f"B. {k}. Sample shapes:", d["metadata_seq"].shape, d["csi_seq"].shape, d["label"].shape)
        print(f"C. {k}. Sample label:", d["metadata_seq"])
        k = k + 1 

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(loader))
    print("D. metadata_seq:", batch["metadata_seq"].shape)  # (B, W, 12)
    print("E. csi_seq:", batch["csi_seq"].shape)            # (B, W, 99)
    print("F. label:", batch["label"].shape)                # (B,)

    import torch.optim as optim

    # Instantiate model
    model = CSILSTMNet(
        csi_input_size=batch["csi_seq"].shape[2],
        meta_input_size=batch["metadata_seq"].shape[2],
        window_size=batch["metadata_seq"].shape[1],
        num_classes=30  # adjust as needed
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    from torch.utils.data import random_split, DataLoader
    import matplotlib.pyplot as plt

    # Split dataset into train and test sets (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    num_epochs = 5
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            csi_seq = batch["csi_seq"]
            meta_seq = batch["metadata_seq"]
            labels = batch["label"].squeeze()
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
                csi_seq = batch["csi_seq"]
                meta_seq = batch["metadata_seq"]
                labels = batch["label"].squeeze()
                outputs = model(csi_seq, meta_seq)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = running_loss / len(test_loader)
        test_losses.append(avg_loss)
        test_accuracies.append(correct / total)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_accuracies[-1]:.4f}")

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_epoch.png")

    # Plot accuracy curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, num_epochs+1), test_accuracies, label="Test Accuracy", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_vs_epoch.png")