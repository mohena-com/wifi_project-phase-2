import tensorflow as tf

def set_gpu_memory_growth():

    tf.debugging.set_log_device_placement(False)

    print("TensorFlow version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    physical_gpus = tf.config.list_physical_devices('GPU')

    if physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(physical_gpus[0], True)
            tf.config.set_visible_devices(physical_gpus[0], 'GPU')
            print(f"Using GPU: {physical_gpus[0]}")
            return '/GPU:0'
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
    else:
        print("No GPU found, using CPU.")
    return '/CPU:0'

# Set device dynamically
device = set_gpu_memory_growth()
print(f"Device: {device}")
# Use device
with tf.device(device):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)

print("Completed matrix multiplication")


