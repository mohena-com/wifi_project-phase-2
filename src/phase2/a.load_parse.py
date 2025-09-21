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
        columns = reader.fieldnames
        complex_cols = [col for col in columns if col.startswith('csi_')]
        numeric_cols = [col for col in columns if not col.startswith('csi_')]

        # Build new column names: real columns + csi_xxx_r + csi_xxx_i
        new_columns = numeric_cols.copy()
        for col in complex_cols:
            new_columns.append(f"{col}_r")
            new_columns.append(f"{col}_i")

        X = []
        for row in reader:
            feature_row = []
            # Add real columns
            for col in numeric_cols:
                feature_row.append(float(row[col]))
            # Add real and imag parts for each complex column
            for col in complex_cols:
                val = parse_complex(row[col])
                feature_row.append(val.real)
                feature_row.append(val.imag)
            X.append(feature_row)
        X = np.array(X)
    return X, new_columns

# Usage:
X, new_columns = load_csv_as_batch_split_complex(
    '/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset/Environment 1/Subject 01/E1_S01_C01_A01_T01.csv'
)
print(X.shape)
print(new_columns)
print(X[0])