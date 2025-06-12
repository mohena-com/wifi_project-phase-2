from flask import Flask, request, jsonify
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import time
import os
import sys
import warnings

from config_reader import ConfigReader  # Ensure this is accessible

app = Flask(__name__)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'real_time_inference_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return log_file


def load_best_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def preprocess_csi_data(csi_data, input_shape):
    csi_data = csi_data.reshape(-1, *input_shape)
    csi_data = csi_data.astype(np.float32)
    return csi_data


def predict_activity(model, csi_data):
    predictions = model.predict(csi_data, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    return predicted_class, confidence


def find_best_model(model_save_dir):
    model_files = list(Path(model_save_dir).glob('best_model_fold_*.keras'))
    if not model_files:
        raise FileNotFoundError("No model files found")
    fold_numbers = [int(str(f).split('_')[-1].split('.')[0]) for f in model_files]
    best_fold = max(fold_numbers)
    return model_save_dir / f'best_model_fold_{best_fold}.keras'


@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Request input
        base_path = request.json['base_path']
        project_name = request.json['project_name']
        config_file = request.json['config_file']

        config = ConfigReader(f"{base_path}/{project_name}/config/{config_file}")
        input_shape = config.get_tuple('input_shape')
        model_dir = Path(config.get('model_save_path'))

        setup_logging(f"{base_path}/logs")

        # Load model
        best_model_path = find_best_model(model_dir)
        model = load_best_model(str(best_model_path))

        # Load test data
        test_data_path = Path(base_path) / project_name / 'sample_test_data/test_data.csv'
        df = pd.read_csv(test_data_path)
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        # Predict
        results = []
        activity_labels = ["Walking", "Running", "Sitting", "Standing", "Lying",
                           "Climbing Up", "Climbing Down", "Jumping", "Falling", "Idle"]
        correct = 0
        for i, (sample, label) in enumerate(zip(features, labels)):
            processed = preprocess_csi_data(sample.reshape(1, -1), input_shape)
            pred_class, conf = predict_activity(model, processed)
            result = {
                "sample": i + 1,
                "true_label": activity_labels[int(label)],
                "predicted_label": activity_labels[pred_class[0]],
                "confidence": float(conf[0]),
                "correct": pred_class[0] == int(label)
            }
            if result["correct"]:
                correct += 1
            results.append(result)
            time.sleep(0.1)  # mimic real-time

        accuracy = (correct / len(features)) * 100
        return jsonify({
            "total_samples": len(features),
            "correct_predictions": correct,
            "accuracy": accuracy,
            "results": results
        })

    except Exception as e:
        logging.error(f"Error in /infer: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
