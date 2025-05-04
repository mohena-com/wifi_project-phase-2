import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path('/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'data_processing_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(stream=sys.stdout)  # Console handler
        ]
    )
    
    # Set console handler to only show WARNING and above
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Get the root logger and remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handlers
    root_logger.addHandler(logging.FileHandler(log_file))
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

# Setup logging
log_file = setup_logging()

# ---- Load CSI features ----
features_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/data/CSI_data_S01.npz'
csi_data = np.load(features_path)['arr_0']  # shape: (num_samples, num_features)
logging.info(f"Loaded CSI features from: {features_path}")

# ---- Load activity labels ----
labels_npz_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/data/output/label/CSI_label_S01.npz'
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

csv_output_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/program_output/mapped_data.csv'
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
df.to_csv(csv_output_path, index=False)
logging.info(f"CSV file saved to: {csv_output_path}")

