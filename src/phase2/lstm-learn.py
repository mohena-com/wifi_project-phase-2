import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Sample data creation
data = np.array([[i for i in range(10)] for _ in range(100)])
print(f'data : {data}')
X, y = data[:, :-1], data[:, -1]

print(f'X shape[0] : {X.shape[0]}, X shape[1] : {X.shape[1]}, y shape : {y.shape}')
# Reshape input to (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(X)
print(y)
# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(9, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, verbose=2)

# Making a prediction
test_input = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15])
test_input = test_input.reshape((1, 9, 1))
predicted_value = model.predict(test_input, verbose=0)
print(f'Predicted value: {predicted_value}')
print(f'Predicted value: {predicted_value[0][0]}')
