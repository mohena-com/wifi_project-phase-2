import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample data creation (replace with your CSV or dataset)
data = [266, 145, 183, 119, 180, 168, 231, 224, 192, 122, 167, 158, 161, 172, 168, 181, 183, 218, 230, 242, 209, 191, 172, 194]
data = np.array(data).astype('float32').reshape(-1, 1)
print(f"Original data shape: {data.shape}")
print(f"NP array data : {data}")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
print(f"Scaler: {scaler}")
dataset = scaler.fit_transform(data)
print(f"dataset: {dataset}")


# Prepare the data in X=t and y=t+1 format
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 3
X, y = create_dataset(dataset, look_back)
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X: {X}, y: {y}")

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, time_steps, features]
print(f"Reshaped X shape: {X.shape}")
print(f"Reshaped X data : {X.shape}")


# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=200, batch_size=1, verbose=1)

# Make predictions (example)
pred = model.predict(X)
pred_inversed = scaler.inverse_transform(pred)
print("First few predictions (in original sales scale):", pred_inversed.flatten())
