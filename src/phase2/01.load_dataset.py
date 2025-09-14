import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# one time execution of this file is needed for creating .npy files.
# .npy files are created for X and y variables separately
# ---------------------------------------------------------------------
def parse_complex(s):
    # Converts '15+15i' or '15-15i' to Python complex
    s = s.replace('+-', '-')
    s = s.replace('-+', '-')
    s = s.replace('i', 'j')
    return complex(s)

def load_csv_as_batch(file, look_back):
    """
    Loads a CSV file and converts CSI values into normalized magnitudes.
    Builds look_back-length windows for temporal modeling.
    Returns: (X_batch, y_batch)
    """
    with open(file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    data = []
    for line in lines:
        row = line.strip().split(',')
        row_complex = [parse_complex(val) for val in row if val]
        data.append(row_complex)
    data = np.array(data)
    if data.size == 0:
        return np.zeros((1, look_back, 1)), np.zeros((1,))
    data_mag = np.abs(data)
    # Normalize
    data_mag = (data_mag - np.min(data_mag)) / (np.max(data_mag) - np.min(data_mag) + 1e-8)
    X_batch, y_batch = [], []
    for i in range(len(data_mag) - look_back):
        X_batch.append(data_mag[i:i+look_back])
        y_batch.append(np.mean(data_mag[i+look_back]))  # placeholder target
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    if X_batch.size == 0:
        return np.zeros((1, look_back, 1)), np.zeros((1,))
    return X_batch, y_batch

# ----------------------------
# Real+Imag parser (for ResNet+LSTM with 2 channels)
# ----------------------------
def parse_complex_str(s):
    if pd.isna(s): return 0+0j
    s = str(s).strip().replace(' ', '').replace('i', 'j').strip("[]()")
    try:
        return complex(s)
    except:
        m = re.match(r'([+-]?\d*\.?\d*)([+-]\d*\.?\d*j)', s)
        if m:
            return complex(m.group(1) + m.group(2))
        return 0+0j

def parse_csi_file(path: Path):
    df = pd.read_csv(path, engine="python")
    csi_cols = [c for c in df.columns if c.lower().startswith("csi")]
    rows, num = len(df), len(csi_cols)
    csi_matrix = np.zeros((rows, num), dtype=np.complex128)
    for i, c in enumerate(csi_cols):
        col = df[c].astype(str).values
        parsed = [parse_complex_str(x) for x in col]
        csi_matrix[:, i] = parsed
    # stack real+imag → shape (N, 180)
    csi_realimag = np.concatenate([csi_matrix.real, csi_matrix.imag], axis=1)
    return csi_realimag

def parse_csi_file_with_metadata(path: Path):
    # Load the whole CSV file
    df = pd.read_csv(path, engine="python")
    
    # Extract the CSI columns
    csi_cols = [c for c in df.columns if c.lower().startswith("csi")]
    rows, num = len(df), len(csi_cols)

    # Parse CSI columns into complex matrix
    csi_matrix = np.zeros((rows, num), dtype=np.complex128)
    for i, c in enumerate(csi_cols):
        col = df[c].astype(str).values
        parsed = [parse_complex_str(x) for x in col]
        csi_matrix[:, i] = parsed
        print(f"Parsed real {parsed.real}: {parsed.imag}: {parsed.imaginary} ...")

    # Stack real and imaginary parts
    csi_realimag = np.concatenate([csi_matrix.real, csi_matrix.imag], axis=1)

    # Extract other metadata columns as numpy arrays (if they exist)
    meta_columns = ['timestamp_low', 'bfee_count', 'Nrx', 'Ntx', 'rssi_a', 'rssi_b', 'rssi_c', 
                    'noise', 'agc', 'perm_1', 'perm_2', 'perm_3', 'rate']
    meta_data = {}
    for col in meta_columns:
        if col in df.columns:
            meta_data[col] = df[col].values
        else:
            meta_data[col] = None  # or np.zeros(rows) if preferred

    # Return dictionary with both metadata and CSI data
    return {
        'metadata': meta_data,     # dict of numpy arrays for metadata fields
        'csi_realimag': csi_realimag  # numpy array with real and imag CSI values
    }


# ----------------------------
# Dataset builder for nested folder structure
# ----------------------------
def build_gait_dataset(base_dir, save_dir="dataset_out"):
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y = [], []
    label_map = {}
    label_counter = 0

    # Traverse environments and subjects
    for env_dir in sorted(base_dir.glob("Environment*")):
        for subj_dir in sorted(env_dir.glob("Subject*")):
            subj_name = subj_dir.name  # e.g. "Subject 1"
            if subj_name not in label_map:
                label_map[subj_name] = label_counter
                label_counter += 1

            # Pick only gait (C03) files
            for f in subj_dir.glob("*.csv"):
                if "C03" not in f.name:
                    continue
                try:
                    arr = parse_csi_file_with_metadata(f) #parse_csi_file(f)
                    X.append(arr)
                    y.append(label_map[subj_name])
                    print(f"Parsed {f}: , label={label_map[subj_name]},Arr={arr}")
                except Exception as e:
                    print(f"❌ Error parsing {f}: {e}")

    # Save combined dataset
    np.save(save_dir / "X.npy", np.array(X, dtype=object))  # object array (var-length sequences)
    np.save(save_dir / "y.npy", np.array(y))
    with open(save_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n✅ Dataset built:")
    print(f"  Total files parsed: {len(X)}")
    print(f"  Subjects: {label_map}")
    print(f"  Saved to: {save_dir}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    build_gait_dataset(
        base_dir="/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset",
        save_dir="/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset_gait"
    )


