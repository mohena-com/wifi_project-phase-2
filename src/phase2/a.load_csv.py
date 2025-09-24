import csv
import numpy as np

def parse_complex(s):
    s = s.replace('+-', '-')
    s = s.replace('-+', '-')
    s = s.replace('i', 'j')
    return complex(s)

def load_csv_as_batch_split_complex(filename):
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        S_num, A_num = extract_S_A_numbers(filename)
 
        columns = reader.fieldnames
        complex_cols = [col for col in columns if col.startswith('csi_')]
        numeric_cols = [col for col in columns if not col.startswith('csi_') and not col.startswith('rate') and not col.startswith('noise')]
        
        used_columns = []
        used_columns.extend(numeric_cols)
        for col in complex_cols:
            used_columns.append(col)  # or append f"{col}_avg" if you want to indicate averaging


        y = []
        X = []
        for row in reader:
            feature_row = []
            target_row = []
            # Add real columns
            for col in numeric_cols:
                feature_row.append(float(row[col]))
            # Add real and imag parts for each complex column
            for col in complex_cols:
                val = parse_complex(row[col])
                feature_row.append(abs(val))  # storing magnituede of complex numbers abs(val)
            target_row.append(S_num)
            target_row.append(A_num)
  
            X.append(feature_row)
            y.append(target_row)
        X = np.array(X)
        y = np.array(y)
    return X, y, used_columns

import re

def extract_S_A_numbers(filename):
    """
    Extracts the number after 'S' and 'A' in the filename.
    Example: 'E1_S01_C01_A03_T01.csv' -> S: 01, A: 03
    Returns (S_number, A_number) as strings.
    """
    match = re.search(r'S(\d+).*A(\d+)', filename)
    if match:
        S_num = match.group(1)
        A_num = match.group(2)
        return S_num, A_num
    return None, None


 
import os

def load_all_C03_files(base_dir):
    all_X = []
    all_y = []
    all_columns = None
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "C03" in file and file.endswith(".csv"):
                print("Loading:", file)
                file_path = os.path.join(root, file)
                X, y, columns = load_csv_as_batch_split_complex(file_path)
                print("Loaded shape:", X.shape, y.shape)
                all_X.append(X)
                all_y.append(y)
                print("Current total shape:", 
                      (np.vstack(all_X).shape if all_X else (0,)), 
                      (np.vstack(all_y).shape if all_y else (0,)))

                if all_columns is None:
                    all_columns = columns
    if all_X:
        all_X = np.vstack(all_X)
    else:
        all_X = np.array([])

    if all_y:
        all_y = np.vstack(all_y)
    else:
        all_y = np.array([])
            
    return all_X, all_y, all_columns

import matplotlib.pyplot as plt

def plot_boxplot_for_features(X, columns, start_col=12, end_col=50):
    """
    Plots a boxplot for features in X from start_col to end_col (exclusive).
    """
    if X.ndim != 2 or X.shape[0] == 0:
        print("X is empty or not 2-dimensional. Cannot plot boxplot.")
        return
    X_sel = X[:, start_col:end_col]
    columns_sel = columns[start_col:end_col]
    plt.figure(figsize=(16, 6))
    plt.boxplot(X_sel, vert=False, labels=columns_sel)
    plt.title(f"Boxplot of Features {start_col+1} to {end_col}")
    plt.xlabel("Value")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("feature_boxplots_selected.png")

base_dir = "/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/temp"
# Usage:
#base_dir = "/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset"
X_C03, y_C03, columns_C03 = load_all_C03_files(base_dir)
#plot_boxplot_for_features(X_C03, columns_C03, start_col=12, end_col=50)
 
print(X_C03.shape)
print(columns_C03)
print(X_C03[0])

'''
s = []
s.append('2+27i')
s.append('-5+26i')
s.append('26+-6i')
s.append('24+10i')
s.append('17+-21i')
s.append('26+4i')
s.append('-14+20i')
s.append('16+23i')
s.append('-2+-27i')
s.append('15+-17i')
s.append('-27+2i')
s.append('6+-26i')

for k in s:
    print(k, "->", parse_complex(k), " Real:", parse_complex(k).real, " Imag:", parse_complex(k).imag, " Avg:", (parse_complex(k).real + parse_complex(k).imag)/2 ) 

# Example usage:
filename = "E1_S01_C01_A03_T01.csv"
S_num, A_num = extract_S_A_numbers(filename)
print("S:", S_num, "A:", A_num)

'''