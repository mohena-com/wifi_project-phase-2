import tensorflow as tf
import sys
from config_reader import ConfigReader
import logging
from distutils.util import strtobool
from pathlib import Path
from datetime import datetime
import os

#python environment.py /Sanjeev/VNIT_CLASSES/FINAL_PROJECT wifi_project har_config.properties False

def parse_args():
    """Parse command-line arguments and return as tuple."""
    argv_1 = sys.argv[1] if len(sys.argv) > 1 else None
    argv_2 = sys.argv[2] if len(sys.argv) > 2 else None
    argv_3 = sys.argv[3] if len(sys.argv) > 3 else None
    argv_4 = sys.argv[4] if len(sys.argv) > 4 else None
    return argv_1, argv_2, argv_3, argv_4

def setup_environment(argv_1, argv_2, argv_3, argv_4):
    """Setup environment and detect if running in Google Colab."""
    
    base_path = f"{argv_1}/{argv_2}"
    config_path = f"{base_path}/config/{argv_3}"
    IN_COLAB = bool(strtobool(argv_4))
    logging.info("Running in Google Colab" if IN_COLAB else "Running locally")
    logging.info(f'Configuration loaded from: {config_path}')
    config = ConfigReader(config_path)
    base_path = config.get_path('data_set_path')

    # Set TensorFlow default float type for memory efficiency
    tf.keras.backend.set_floatx('float32')

    # Check for GPU using TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info("GPU is available")
        logging.info(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            logging.info(f"- {gpu.name}")
        # Configure GPU memory growth
        if config.get_bool('gpu_memory_growth'):
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("GPU memory growth enabled")
            except RuntimeError as e:
                logging.error(f"Error setting GPU memory growth: {e}")
    else:
        logging.warning("No GPU devices found")
    print(f'    ENVIRONMENT SETUP COMPLETE')
    return IN_COLAB, base_path, config

def setup_logging(argv_1):
    """Setup logging configuration"""
    log_path = f"{argv_1}/logs"
    log_dir = Path(log_path).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'har_training_{timestamp}.log'

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )

    # Configure TensorFlow logging
    tf.get_logger().setLevel(logging.ERROR)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TF messages

    logging.info(f"Logging initialized. Log file: {log_file}")
    print(f'    LOG SETUP COMPLETE')
    return log_file

if __name__ == "__main__":
    argv_1, argv_2, argv_3, argv_4 = parse_args()
    setup_logging(argv_1)
    setup_environment(argv_1, argv_2, argv_3, argv_4)

#python environment.py /content/drive/MyDrive/Intellipaat-sem3/wifi_project/FINAL_PROJECT/wifi_project/config/har_config.properties True
#python environment.py /Sanjeev/VNIT_CLASSES/FINAL_PROJECT/wifi_project/config/har_config.properties False

#python environment.py /Sanjeev/VNIT_CLASSES/FINAL_PROJECT/wifi_project  har_config.properties False
#python environment.py /content/drive/MyDrive/Intellipaat-sem3/wifi_project/FINAL_PROJECT/wifi_project har_config.properties True

