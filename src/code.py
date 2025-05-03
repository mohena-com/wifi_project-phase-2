import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---- Load CSI features ----
features_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/data/CSI_data_S01.npz'
csi_data = np.load(features_path)['arr_0']  # shape: (num_samples, num_features)

# ---- Load activity labels ----
labels_npz_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/data/output/label/CSI_label_S01.npz'
label_data = np.load(labels_npz_path)
print("Label NPZ keys:", label_data.files)  # Should show ['arr_0']

activity_labels = label_data['arr_0']  # shape should match csi_data.shape[0]
print("Labels shape:", activity_labels.shape)
print("Sample labels:", np.unique(activity_labels))
print("Feature samples:", csi_data.shape[0])
print("Label samples: ", activity_labels.shape[0])

# ---- Encode string labels to numeric ----
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(activity_labels)
print("Label encoding mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# ---- Map each data row to its label and print samples ----
mapped_data = list(zip(csi_data, y_encoded))
for i in range(5):
    print(f"Feature {i}: {mapped_data[i][0]}, Label: {mapped_data[i][1]}")

# ---- Save to CSV ----
feature_columns = [f'feature_{i+1}' for i in range(csi_data.shape[1])]
df = pd.DataFrame(csi_data, columns=feature_columns)
df['label'] = y_encoded

csv_output_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/program_output/mapped_data.csv'
df.to_csv(csv_output_path, index=False)
print(f"CSV file saved to: {csv_output_path}")

