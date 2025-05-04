import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import math
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_percentage_error
import seaborn as sns
import pandas as pd
import argparse
import sys
from config_reader import ConfigReader
from datetime import datetime
import time
import logging
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow CPU info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure logging
def setup_logging(config):
    """Setup logging configuration"""
    # Get log directory from config and convert to absolute path
    log_dir = Path(config.get('log_save_path'))
    log_dir = log_dir.resolve()  # Convert to absolute path
    
    # Create logs directory if it doesn't exist
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Log directory created/verified at: {log_dir}")
    except Exception as e:
        logging.error(f"Failed to create log directory: {e}")
        # Fallback to a local logs directory if the configured one fails
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.warning(f"Using fallback log directory: {log_dir}")
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'har_training_{timestamp}.log'
    
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
    
    # Configure TensorFlow logging
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.addHandler(logging.FileHandler(log_file))
    tf_logger.setLevel(logging.INFO)
    
    # Configure training progress logging
    progress_logger = logging.getLogger('training_progress')
    progress_logger.addHandler(logging.FileHandler(log_file))
    progress_logger.setLevel(logging.INFO)
    
    # Configure TensorFlow CPU feature guard messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

def setup_environment(config):
    """Setup environment and detect if running in Google Colab."""
    try:
        from google.colab import drive
        IN_COLAB = True
        logging.info("Running in Google Colab")
        
        # Mount Google Drive
        drive.mount('/content/drive')
        
        # Set base path for Colab
        base_path = config.get_path('colab_data_path')
        
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
                
    except ImportError:
        IN_COLAB = False
        logging.info("Running locally")
        base_path = config.get_path('local_data_path')
        
        # Check for GPU locally
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info("GPU is available")
            logging.info(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                logging.info(f"- {gpu.name}")
        else:
            logging.warning("No GPU devices found")
    
    return IN_COLAB, base_path

def load_csi_data(base_path, in_colab, config):
    """Load all CSI data and labels from the dataset directory."""
    logging.info(f"Loading data from base path: {base_path}")
    
    # Convert base_path to absolute path if it's not already
    base_path = Path(base_path).resolve()
    logging.info(f"Resolved base path: {base_path}")
    
    # Define data and label directories
    data_dir = base_path / 'data'
    label_dir = base_path / 'data' / 'output' / 'label'
    
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Label directory: {label_dir}")
    
    # Check if directories exist
    if not data_dir.exists():
        logging.error(f"Error: Data directory does not exist: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if not label_dir.exists():
        logging.error(f"Error: Label directory does not exist: {label_dir}")
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    all_features = []
    all_labels = []
    
    # Get the range from config
    data_range = config.get_int('data_set_range')
    logging.info(f"Loading data for subjects 1 to {data_range}")
    
    # Load each subject's data
    for subject in range(1, data_range + 1):
        subject_str = f'S{subject:02d}'
        features_path = data_dir / f'CSI_data_{subject_str}.npz'
        labels_path = label_dir / f'CSI_label_{subject_str}.npz'
        
        logging.info(f"\nChecking for subject {subject_str}:")
        logging.info(f"Features path: {features_path}")
        logging.info(f"Labels path: {labels_path}")
        
        if features_path.exists() and labels_path.exists():
            try:
                csi_data = np.load(str(features_path))['arr_0']
                activity_labels = np.load(str(labels_path))['arr_0']
                
                # Log NPZ file information
                logging.info(f"Label NPZ keys: {list(np.load(str(labels_path)).files)}")
                logging.info(f"Labels shape: {activity_labels.shape}")
                logging.info(f"Sample labels: {np.unique(activity_labels)}")
                logging.info(f"Feature samples: {csi_data.shape[0]}")
                logging.info(f"Label samples: {activity_labels.shape[0]}")
                
                # Log sample feature data
                logging.info("\nSample feature data:")
                for i in range(min(5, len(csi_data))):  # Log first 5 samples
                    logging.info(f"Feature {i}: {csi_data[i]}")
                
                all_features.append(csi_data)
                all_labels.append(activity_labels)
            except Exception as e:
                logging.error(f"Error loading data for subject {subject_str}: {str(e)}")
                continue
        else:
            logging.warning(f"Files not found for subject {subject_str}")
            if not features_path.exists():
                logging.warning(f"Missing features file: {features_path}")
            if not labels_path.exists():
                logging.warning(f"Missing labels file: {labels_path}")
    
    if not all_features:
        raise ValueError("No data files were found. Please check the data directory structure and file names.")
    
    # Concatenate all data
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    logging.info(f"\nFinal concatenated data shape: {X.shape}")
    logging.info(f"Final concatenated labels shape: {y.shape}")
    
    return X, y

# ResNet-1D Block
def resnet_block(inputs, filters, kernel_size=3, stride=1):
    """Create a ResNet block with proper shape handling."""
    # Main path
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Shortcut path
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    # Add shortcut to main path
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Attention Mechanism
def attention_layer(inputs):
    attention = layers.Dense(1, activation='tanh')(inputs)
    attention = layers.Flatten()(attention)
    attention_weights = layers.Activation('softmax')(attention)
    attention_weights = layers.Reshape((inputs.shape[1], 1))(attention_weights)
    return layers.Multiply()([inputs, attention_weights])

# Build Hybrid Model
def create_hybrid_model(input_shape, num_classes, config):
    """Build the hybrid model using configuration parameters."""
    inputs = layers.Input(shape=input_shape)

    # ResNet-1D blocks
    filters = config.get_list('resnet_filters', type_func=int)
    kernel_sizes = config.get_list('kernel_size', type_func=int)
    strides = config.get_list('stride', type_func=int)
    
    # Initial convolution
    x = layers.Conv1D(filters[0], kernel_sizes[1], strides=strides[1], padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=strides[1], padding='same')(x)

    # ResNet blocks
    for i, filter_size in enumerate(filters):
        if i == 0:
            x = resnet_block(x, filter_size, kernel_size=kernel_sizes[0], stride=strides[0])
        else:
            x = resnet_block(x, filter_size, kernel_size=kernel_sizes[0], stride=strides[1])

    # BiLSTM layer
    lstm_units = config.get_int('lstm_units')
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    # Attention mechanism
    x = attention_layer(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(config.get_float('dropout_rate'))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

# Mixup Augmentation
def mixup_data(x, y, alpha=0.2):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x + (1 - lam) * tf.gather(x, indices)
    y_mixed = lam * y + (1 - lam) * tf.gather(y, indices)

    return x_mixed, y_mixed

# Cosine Learning Rate Decay
def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, learning_rate_base=3e-4):
    if epoch < warmup_epochs:
        return learning_rate_base * ((epoch + 1) / warmup_epochs)

    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return learning_rate_base * 0.5 * (1 + math.cos(math.pi * progress))

# Custom callback for logging training progress
class TrainingProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        # Create a dedicated logger for training progress
        self.logger = logging.getLogger('training_progress')
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        # Add only file handler
        self.logger.addHandler(logging.FileHandler(log_file))
        self.logger.setLevel(logging.INFO)
        # Disable propagation to root logger
        self.logger.propagate = False
        
        self.last_log_time = time.time()
        self.log_interval = 5  # seconds between progress logs
        self.batch_count = 0
        self.total_batches = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting Epoch {epoch + 1}/5 at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*50}")
        self.batch_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.logger.info(f"Epoch {epoch + 1} - loss: {logs.get('loss', 0):.4f} - accuracy: {logs.get('accuracy', 0):.4f} - val_loss: {logs.get('val_loss', 0):.4f} - val_accuracy: {logs.get('val_accuracy', 0):.4f}")
    
    def on_train_batch_begin(self, batch, logs=None):
        self.total_batches = logs.get('size', 0) if logs else 0
    
    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        self.batch_count += 1
        
        # Log first batch
        if batch == 0:
            self.logger.info(f"Batch {batch} - loss: {logs.get('loss', 0):.4f} - accuracy: {logs.get('accuracy', 0):.4f} - categorical_accuracy: {logs.get('categorical_accuracy', 0):.4f}")
            self.last_log_time = current_time
            return
        
        # Log progress every log_interval seconds
        if (current_time - self.last_log_time) >= self.log_interval:
            progress = f"{self.batch_count}/{self.total_batches}"
            eta = self._calculate_eta(batch, current_time)
            self.logger.info(f"Batch {batch} - loss: {logs.get('loss', 0):.4f} - accuracy: {logs.get('accuracy', 0):.4f} - categorical_accuracy: {logs.get('categorical_accuracy', 0):.4f}")
            self.last_log_time = current_time
    
    def _calculate_eta(self, current_batch, current_time):
        if current_batch == 0:
            return "N/A"
        elapsed_time = current_time - self.last_log_time
        remaining_batches = self.total_batches - current_batch
        eta_seconds = (elapsed_time / current_batch) * remaining_batches
        minutes = int(eta_seconds // 60)
        seconds = int(eta_seconds % 60)
        return f"{minutes}:{seconds:02d}"

# Training Function
def train_model(x_train, y_train, x_val, y_val, model, fold, config, epochs=5, batch_size=64):
    """Train the model with the given data and parameters."""
    # Prepare callbacks
    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, epochs))
    
    # Create model save directory
    model_save_dir = Path(config.get('model_save_path'))
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Only save best models if configured
    callbacks = [lr_scheduler]
    if config.get_bool('save_fold_models', default=True):
        checkpoint = ModelCheckpoint(
            str(model_save_dir / f'best_model_fold_{fold + 1}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0  # Don't print checkpoint messages
        )
        callbacks.append(checkpoint)
    
    # Create training progress logger
    log_file = Path(config.get('log_save_path')) / f'training_progress_fold_{fold + 1}.log'
    progress_logger = TrainingProgressLogger(log_file)
    callbacks.append(progress_logger)

    # Training with mixup
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=0  # Don't print training progress
    )

    return history

def select_best_model(fold_histories, config):
    """Select the best model across all folds based on validation accuracy."""
    best_fold = 0
    best_val_accuracy = 0.0
    
    for fold, history in enumerate(fold_histories):
        max_val_accuracy = max(history['val_accuracy'])
        if max_val_accuracy > best_val_accuracy:
            best_val_accuracy = max_val_accuracy
            best_fold = fold
    
    model_save_dir = Path(config.get('model_save_path'))
    best_model_path = model_save_dir / f'best_model_fold_{best_fold + 1}.keras'
    
    if best_model_path.exists():
        logging.info(f"Selected best model from fold {best_fold + 1} with validation accuracy: {best_val_accuracy:.4f}")
        return best_model_path
    else:
        logging.warning("No model files found. Please ensure 'save_fold_models' is enabled in config.")
        return None

# Plotting Function
def plot_training_results(fold_histories, config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for i, history in enumerate(fold_histories):
        ax1.plot(history['accuracy'], label=f'Fold {i + 1}')
        ax2.plot(history['val_accuracy'], label=f'Fold {i + 1}')

    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(config.get('plot_save_path')) / 'training_results.png')
    plt.close()

def plot_evaluation_metrics(fold_histories, y_true, y_pred, encoder, config, X_data=None):
    """Plot comprehensive evaluation metrics and visualizations."""
    # Create a directory for plots
    os.makedirs(config.get('plot_save_path'), exist_ok=True)
    
    # 1. Plot Epoch vs Metrics
    plt.figure(figsize=(15, 10))
    metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        for fold, history in enumerate(fold_histories):
            plt.plot(history[metric], label=f'Fold {fold + 1}')
        plt.title(f'Epoch vs {metric.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
    plt.tight_layout()
    plt.savefig(Path(config.get('plot_save_path')) / 'epoch_metrics.png')
    plt.close()
    
    # 2. Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(config.get('plot_save_path')) / 'confusion_matrix.png')
    plt.close()
    
    # 3. Plot Class Distribution
    plt.figure(figsize=(12, 6))
    class_counts = pd.Series(y_true).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Activity Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(config.get('plot_save_path')) / 'class_distribution.png')
    plt.close()
    
    # 4. Plot Feature Distribution (Box Plot) - Only if X_data is provided
    if X_data is not None:
        plt.figure(figsize=(15, 8))
        feature_data = pd.DataFrame(X_data.reshape(-1, 90))
        sns.boxplot(data=feature_data.iloc[:, :10])  # First 10 features
        plt.title('Feature Distribution (First 10 Features)')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(Path(config.get('plot_save_path')) / 'feature_distribution.png')
        plt.close()
        
        # 5. Plot Correlation Heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = feature_data.corr()
        sns.heatmap(correlation_matrix.iloc[:10, :10], cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap (First 10 Features)')
        plt.tight_layout()
        plt.savefig(Path(config.get('plot_save_path')) / 'correlation_heatmap.png')
        plt.close()
    
    # 6. Plot Learning Curves
    plt.figure(figsize=(12, 6))
    for fold, history in enumerate(fold_histories):
        plt.plot(history['accuracy'], label=f'Training Fold {fold + 1}')
        plt.plot(history['val_accuracy'], label=f'Validation Fold {fold + 1}', linestyle='--')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(config.get('plot_save_path')) / 'learning_curves.png')
    plt.close()
    
    # 7. Calculate and Plot Additional Metrics
    metrics_data = {
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_data.keys(), metrics_data.values())
    plt.title('Additional Evaluation Metrics')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(Path(config.get('plot_save_path')) / 'additional_metrics.png')
    plt.close()

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'har_config.properties'
    config = ConfigReader(config_path)
    logging.info(f'Configuration loaded from: {config_path}')
    
    # Setup logging
    log_file = setup_logging(config)
    logging.info(f'Logging started. Log file: {log_file}')
    
    # Setup environment and get base path
    IN_COLAB, base_path = setup_environment(config)
    logging.info(f'Environment setup - IN_COLAB: {IN_COLAB}, base_path: {base_path}')
    
    # Load and preprocess data
    X, y = load_csi_data(base_path, IN_COLAB, config)
    logging.info(f'Data loaded - X shape: {X.shape}, y shape: {y.shape}')
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = config.get_int('num_classes')
    logging.info(f'Labels encoded - Number of classes: {num_classes}')
    
    # Log label encoding mapping
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    logging.info(f'Label encoding mapping: {label_mapping}')
    
    # Log sample features after encoding
    logging.info("\nSample features after encoding:")
    for i in range(min(5, len(X))):  # Log first 5 samples
        logging.info(f"Feature {i}: {X[i]}, Label: {y_encoded[i]}")
    
    # Reshape data for the model
    input_shape = config.get_tuple('input_shape')
    X = X.reshape(-1, *input_shape)
    y_data = tf.keras.utils.to_categorical(y_encoded)
    logging.info(f'Data reshaped - New X shape: {X.shape}')
    
    # Split data
    test_size = config.get_float('test_size')
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_data, test_size=test_size, random_state=config.get_int('random_state'), 
        stratify=y_encoded
    )
    logging.info(f'Data split - Train/Val shape: {X_train_val.shape}, Test shape: {X_test.shape}')
    
    # Initialize cross-validation
    kf = KFold(n_splits=config.get_int('n_splits'), 
               shuffle=True, 
               random_state=config.get_int('random_state'))
    fold_histories = []
    test_metrics = []
    
    # Create directories for saving
    os.makedirs(config.get('model_save_path'), exist_ok=True)
    os.makedirs(config.get('plot_save_path'), exist_ok=True)
    logging.info('Created directories for saving models and plots')
    
    # Start total training time
    total_start_time = time.time()
    logging.info(f"Starting total training process at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        # Start fold training time
        fold_start_time = time.time()
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting Fold {fold + 1}/{config.get_int('n_splits')} at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"{'='*50}")
        
        x_train, x_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Create and compile model
        model = create_hybrid_model(input_shape, num_classes, config)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=config.get_float('learning_rate')),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy']
        )
        logging.info('Model created and compiled')
        
        # Train the model
        history = train_model(x_train, y_train, x_val, y_val, model, fold, config,
                            epochs=config.get_int('epochs'),
                            batch_size=config.get_int('batch_size'))
        fold_histories.append(history.history)
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test, verbose=0)
        test_metrics.append({
            'fold': fold + 1,
            'test_loss': test_results[0],
            'test_accuracy': test_results[1],
            'test_categorical_accuracy': test_results[2]
        })
        
        # Calculate fold training time
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        logging.info(f"Fold {fold + 1} completed in: {fold_duration:.2f} seconds")
        logging.info(f"Fold {fold + 1} completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Test results - Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, Categorical Accuracy: {test_results[2]:.4f}")
        
        # Save the model
        model_path = Path(config.get('model_save_path')) / f'har_model_fold_{fold + 1}.keras'
        model.save(str(model_path))
        logging.info(f"Model saved to: {model_path}")
    
    # Calculate total training time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"\n{'='*50}")
    logging.info(f"Total training completed in: {total_duration:.2f} seconds")
    logging.info(f"Total training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'='*50}")
    
    # Get predictions for the last fold's model on test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Plot and save results
    plot_training_results(fold_histories, config)
    plot_evaluation_metrics(fold_histories, y_true, y_pred_classes, encoder, config, X_data=X)
    logging.info("Training results plotted and saved")

    # After training all folds
    best_model_path = select_best_model(fold_histories, config)
    if best_model_path:
        logging.info(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main() 