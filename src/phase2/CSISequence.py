import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def parse_complex(s):
    # Converts '15+15i' or '15-15i' to Python complex
    s = s.replace('+-', '-')
    s = s.replace('-+', '-')
    s = s.replace('i', 'j')
    return complex(s)

def load_csv_as_batch(file, look_back):
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
        y_batch.append(np.mean(data_mag[i+look_back]))  # Use mean as target
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    if X_batch.size == 0:
        return np.zeros((1, look_back, 1)), np.zeros((1,))
    X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2]))
    return X_batch, y_batch

# Set paths and parameters
#data_folder = r'C:\Sanjeev\VNIT_CLASSES\VNIT-AAI-SEM4\wifi_csv_net_practice_dataset'
data_folder = r'C:\Sanjeev\VNIT_CLASSES\VNIT-AAI-SEM4\wifi-csi-2gb-dataset'
look_back = 3

# Get all CSV files and split for train/test
all_files = glob.glob(os.path.join(data_folder, '**', '*.csv'), recursive=True)
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Dynamically determine num_features for LSTM input shape
sample_X, _ = load_csv_as_batch(train_files[0], look_back)
look_back = sample_X.shape[1]
num_features = sample_X.shape[2]

# Build model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, num_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model incrementally on each train file
for file in train_files:
    print(f"Training on {file}")
    X_batch, y_batch = load_csv_as_batch(file, look_back)
    if X_batch.size == 0:
        continue
    model.fit(X_batch, y_batch, epochs=1, batch_size=32, verbose=1)
    del X_batch, y_batch  # Free memory

# Evaluate on test files
y_true, y_pred = [], []
for file in test_files:
    X_batch, y_batch = load_csv_as_batch(file, look_back)
    if X_batch.size == 0:
        continue
    preds = model.predict(X_batch)
    y_true.extend(y_batch)
    y_pred.extend(preds.flatten())
    del X_batch, y_batch, preds

mse = mean_squared_error(y_true, y_pred)
print("Test MSE:", mse)

# Plot predictions vs true values
plt.figure(figsize=(10,5))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('True vs Predicted')
plt.savefig('predictions.png')