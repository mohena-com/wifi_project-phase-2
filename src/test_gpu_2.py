import tensorflow as tf
import time

# Check available devices
print("Available Physical Devices:")
for device in tf.config.list_physical_devices():
    print(device)
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
# Ensure operations are explicitly placed on GPU
with tf.device('/GPU:0'):
#with tf.device('/CPU:0'):
    print("Running large matrix multiplication on GPU")
    
    # Create large tensors
    size = 4096*10  # Can increase if RAM allows
    A = tf.random.normal([size, size])
    B = tf.random.normal([size, size])

    # Time the matrix multiplication
    start_time = time.time()
    C = tf.matmul(A, B)
    end_time = time.time()
    

    print(f"Time taken: {end_time - start_time:.2f} seconds")