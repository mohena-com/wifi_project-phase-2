import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Helper: Parse CSI string like '2+23i'
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

# ----------------------------
# Parse one CSV file into (N, 180) array
# ----------------------------
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
# Main dataset builder
# ----------------------------
def build_gait_dataset(data_dir, save_dir="dataset_out"):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # find gait (C03) files
    files = [f for f in data_dir.glob("*.csv") if "C03" in f.name]

    X, y = [], []
    label_map = {}
    label_counter = 0

    for f in files:
        subj_match = re.search(r"S\d+", f.name)
        if not subj_match:
            continue
        subj_id = subj_match.group(0)  # e.g. S01
        if subj_id not in label_map:
            label_map[subj_id] = label_counter
            label_counter += 1

        try:
            arr = parse_csi_file(f)
            X.append(arr)
            y.append(label_map[subj_id])
            print(f"Parsed {f.name}: shape {arr.shape}, label={label_map[subj_id]}")
        except Exception as e:
            print(f"❌ Error parsing {f.name}: {e}")

    # save outputs
    np.save(save_dir / "X.npy", np.array(X, dtype=object))  # object array (var-length seqs)
    np.save(save_dir / "y.npy", np.array(y))
    with open(save_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n✅ Dataset built:")
    print(f"  Files parsed: {len(X)}")
    print(f"  Subjects: {label_map}")
    print(f"  Saved to: {save_dir}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    build_gait_dataset(data_dir="/mnt/data", save_dir="/mnt/data/gait_dataset")
