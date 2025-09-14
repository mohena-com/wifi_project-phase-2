import numpy as np
import json

# Load arrays
X = np.load("/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset_gait/X.npy", allow_pickle=True)
y = np.load("/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset_gait/y.npy")

print("X type:", type(X))
print("X length (#files parsed):", len(X))
print("First entry shape:", X[0].shape)   # shape (N_packets, 180)
print("y shape:", y.shape)
print("Unique labels:", np.unique(y))

# Inspect first sample
print("First sample array snippet:\n", X[0][:5, :10])  # first 5 packets, first 10 features
print("First label:", y[0])

# Load label map
with open("/Users/sanjeev/VNIT/FINAL_PRJ_PHASE2/DATASET/wifi-csi-2gb-dataset_gait/label_map.json") as f:
    label_map = json.load(f)
print("Label map:", label_map)


import matplotlib.pyplot as plt

sample = X[0]               # (N_packets, 180)
plt.imshow(np.abs(sample.T), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("CSI magnitude (Subject " + str(y[0]) + ")")
plt.xlabel("Packets")
plt.ylabel("Subcarrier*Antennas (180)")
plt.savefig('packets.png')
