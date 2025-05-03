import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import math
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_percentage_error
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

# Data loading and preprocessing
def load_csi_data(base_path):
    """Load all CSI data and labels from the dataset directory."""
    data_dir = Path(base_path) / 'DATASET' / 'data'
    label_dir = Path(base_path) / 'DATASET' / 'data' / 'output' / 'label'
    
    print(f"Looking for data in: {data_dir}")
    print(f"Looking for labels in: {label_dir}")
    
    all_features = []
    all_labels = []
    
    # Load each subject's data
    for subject in range(1, 3):
        subject_str = f'S{subject:02d}'
        features_path = data_dir / f'CSI_data_{subject_str}.npz'
        labels_path = label_dir / f'CSI_label_{subject_str}.npz'
        
        print(f"\nChecking for subject {subject_str}:")
        print(f"Features path exists: {features_path.exists()}")
        print(f"Labels path exists: {labels_path.exists()}")
        
        if features_path.exists() and labels_path.exists():
            try:
                csi_data = np.load(features_path)['arr_0']
                activity_labels = np.load(labels_path)['arr_0']
                
                print(f"Loaded data shape: {csi_data.shape}")
                print(f"Loaded labels shape: {activity_labels.shape}")
                
                all_features.append(csi_data)
                all_labels.append(activity_labels)
            except Exception as e:
                print(f"Error loading data for subject {subject_str}: {str(e)}")
                continue
    
    if not all_features:
        raise ValueError("No data files were found. Please check the data directory structure and file names.")
    
    # Concatenate all data
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"\nFinal concatenated data shape: {X.shape}")
    print(f"Final concatenated labels shape: {y.shape}")
    
    return X, y

# ResNet-1D Block
def resnet_block(inputs, filters, kernel_size=3, stride=1):
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(inputs)
    else:
        shortcut = inputs

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
def create_hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # ResNet-1D blocks
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # ResNet blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 256, stride=2)

    # BiLSTM layer
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Attention mechanism
    x = attention_layer(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
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

# Training Function
def train_model(x_train, y_train, x_val, y_val, model, fold, epochs=5, batch_size=64):
    """Train the model with the given data and parameters."""
    # Prepare callbacks
    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, epochs))
    
    checkpoint = ModelCheckpoint(
        f'best_model_fold_{fold + 1}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Training with mixup
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[lr_scheduler, checkpoint],
        verbose=1
    )

    return history

# Plotting Function
def plot_training_results(fold_histories):
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
    plt.savefig('training_results.png')
    plt.close()

def plot_evaluation_metrics(fold_histories, y_true, y_pred, encoder):
    """Plot comprehensive evaluation metrics and visualizations."""
    # Create a directory for plots
    os.makedirs('evaluation_plots', exist_ok=True)
    
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
    plt.savefig('evaluation_plots/epoch_metrics.png')
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
    plt.savefig('evaluation_plots/confusion_matrix.png')
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
    plt.savefig('evaluation_plots/class_distribution.png')
    plt.close()
    
    # 4. Plot Feature Distribution (Box Plot)
    plt.figure(figsize=(15, 8))
    feature_data = pd.DataFrame(X.reshape(-1, 90))
    sns.boxplot(data=feature_data.iloc[:, :10])  # First 10 features
    plt.title('Feature Distribution (First 10 Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('evaluation_plots/feature_distribution.png')
    plt.close()
    
    # 5. Plot Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = feature_data.corr()
    sns.heatmap(correlation_matrix.iloc[:10, :10], cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap (First 10 Features)')
    plt.tight_layout()
    plt.savefig('evaluation_plots/correlation_heatmap.png')
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
    plt.savefig('evaluation_plots/learning_curves.png')
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
    plt.savefig('evaluation_plots/additional_metrics.png')
    plt.close()

def main():
    # Load and preprocess data
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"Base path: {base_path}")
    
    X, y = load_csi_data(base_path)
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class labels: {encoder.classes_}")
    
    # Reshape data for the model - each sample has 90 features
    X = X.reshape(-1, 90, 1)  # Reshape to (samples, timesteps, features)
    y_data = tf.keras.utils.to_categorical(y_encoded)
    
    print(f"Input data shape: {X.shape}")
    print(f"Label data shape: {y_data.shape}")
    
    # Split data into train+val (80%) and test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_data, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain+Val data shape: {X_train_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize 5-fold cross-validation on train+val data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_histories = []
    test_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        print(f"\nTraining Fold {fold + 1}/5")
        
        x_train, x_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Create and compile model
        model = create_hybrid_model((90, 1), num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        history = train_model(x_train, y_train, x_val, y_val, model, fold)
        fold_histories.append(history.history)
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_metrics.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        })
        
        print(f"Test Accuracy (Fold {fold + 1}): {test_accuracy:.4f}")
        print(f"Test Loss (Fold {fold + 1}): {test_loss:.4f}")
        
        # Save the model for each fold
        model.save(f'har_model_fold_{fold + 1}.keras')
    
    # Calculate and print average test metrics
    avg_test_accuracy = np.mean([m['test_accuracy'] for m in test_metrics])
    avg_test_loss = np.mean([m['test_loss'] for m in test_metrics])
    print(f"\nAverage Test Accuracy across folds: {avg_test_accuracy:.4f}")
    print(f"Average Test Loss across folds: {avg_test_loss:.4f}")
    
    # Plot training results
    plot_training_results(fold_histories)
    
    # Plot test metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar([m['fold'] for m in test_metrics], [m['test_accuracy'] for m in test_metrics])
    plt.title('Test Accuracy per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar([m['fold'] for m in test_metrics], [m['test_loss'] for m in test_metrics])
    plt.title('Test Loss per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/test_metrics.png')
    plt.close()
    
    # Get predictions for the last fold's model on test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Plot all evaluation metrics
    plot_evaluation_metrics(fold_histories, y_true, y_pred_classes, encoder)

if __name__ == "__main__":
    main() 