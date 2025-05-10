import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from environment import setup_environment
from environment import setup_logging
import logging

# Setup environment and get base path
IN_COLAB, base_path, config = setup_environment()
print(f'main : config : {config}')
logging.info(f'Environment setup - IN_COLAB: {IN_COLAB}, base_path: {base_path}')
# Setup logging
log_file = setup_logging(config)
logging.info(f'Logging started. Log file: {log_file}')

# ---- Load CSI features ----
features_path = config.get('features_path')
csi_data = np.load(features_path)['arr_0']  # shape: (num_samples, num_features)
logging.info(f"Loaded CSI features from: {features_path}")

# ---- Load activity labels ----
labels_npz_path = config.get('labels_path')
label_data = np.load(labels_npz_path)
logging.info(f"Label NPZ keys: {label_data.files}")  # Should show ['arr_0']

activity_labels = label_data['arr_0']  # shape should match csi_data.shape[0]
logging.info(f"Labels shape: {activity_labels.shape}")
logging.info(f"Sample labels: {np.unique(activity_labels)}")
logging.info(f"Feature samples: {csi_data.shape[0]}")
logging.info(f"Label samples: {activity_labels.shape[0]}")

# ---- Encode string labels to numeric ----
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(activity_labels)
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
logging.info(f"Label encoding mapping: {label_mapping}")

# ---- Map each data row to its label and print samples ----
mapped_data = list(zip(csi_data, y_encoded))
logging.info("\nSample data points:")
for i in range(5):
    logging.info(f"Feature {i}: {mapped_data[i][0]}, Label: {mapped_data[i][1]}")

# ---- Save to CSV ----
feature_columns = [f'feature_{i+1}' for i in range(csi_data.shape[1])]
df = pd.DataFrame(csi_data, columns=feature_columns)
df['label'] = y_encoded

# Get CSV output path from config
csv_output_path = config.get('csv_output_path')
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
df.to_csv(csv_output_path, index=False)
logging.info(f"CSV file saved to: {csv_output_path}")

