import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
import math


# 1. Define the ResNet1D Block
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


# 2. Define Attention Mechanism
def attention_layer(inputs):
    attention = layers.Dense(1, activation='tanh')(inputs)
    attention = layers.Flatten()(attention)
    attention_weights = layers.Activation('softmax')(attention)
    attention_weights = layers.Reshape((inputs.shape[1], 1))(attention_weights)
    return layers.Multiply()([inputs, attention_weights])


# 3. Build the Hybrid Model
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


# 4. Implement Mixup Augmentation
def mixup_data(x, y, alpha=0.2):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    lam = np.random.beta(alpha, alpha)
    x_mixed = lam * x + (1 - lam) * tf.gather(x, indices)
    y_mixed = lam * y + (1 - lam) * tf.gather(y, indices)

    return x_mixed, y_mixed


# 5. Cosine Learning Rate Decay
def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, learning_rate_base=3e-4):
    if epoch < warmup_epochs:
        return learning_rate_base * ((epoch + 1) / warmup_epochs)

    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return learning_rate_base * 0.5 * (1 + math.cos(math.pi * progress))


# 6. Training Function
def train_model(x_train, y_train, x_val, y_val, model, epochs=100, batch_size=32):
    # Prepare callbacks
    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, epochs))

    # Training with mixup
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[lr_scheduler]
    )

    return history


# 7. Main Training Pipeline
def main():
    # Load the data from the existing code
    features_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/data/CSI_data_S01.npz'
    labels_npz_path = '/Sanjeev/VNIT_CLASSES/FINAL_PROJECT/DATASET/data/output/label/CSI_label_S01.npz'

    csi_data = np.load(features_path)['arr_0']
    activity_labels = np.load(labels_npz_path)['arr_0']

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(activity_labels)
    num_classes = len(encoder.classes_)

    # Reshape data for the model (if needed)
    # Assuming CSI data is already in shape (samples, 100, 30)
    x_data = csi_data.reshape(-1, 100, 30)
    y_data = tf.keras.utils.to_categorical(y_encoded)

    # Initialize 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        print(f"\nTraining Fold {fold + 1}/5")

        x_train, x_val = x_data[train_idx], x_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        # Create and compile model
        model = create_hybrid_model((100, 30), num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        history = train_model(x_train, y_train, x_val, y_val, model)
        fold_histories.append(history.history)

        # Save the model for each fold
        model.save(f'har_model_fold_{fold + 1}.h5')

    # Plot training results
    plot_training_results(fold_histories)


# 8. Plotting Function
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


if __name__ == "__main__":
    main()