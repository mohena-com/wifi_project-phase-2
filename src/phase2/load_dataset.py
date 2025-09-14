import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Your tested helpers
# ----------------------------
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
                    arr = parse_csi_file(f)
                    X.append(arr)
                    y.append(label_map[subj_name])
                    print(f"Parsed {f}: shape {arr.shape}, label={label_map[subj_name]}")
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


