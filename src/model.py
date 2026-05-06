"""
Wildlife Image Classification — Training script
================================================

Trains a 4-block CNN on a 6-class wildlife image dataset
(elephant, giraffe, leopard, rhino, tiger, zebra).

Adapted from the INF7370 course skeleton (Université du Québec à Montréal,
Winter 2026) and refactored for general use.

Author: Majda Bouhou
"""

import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Conv2D, MaxPooling2D, Input, BatchNormalization,
    Activation, Dropout, Dense, GlobalAveragePooling2D,
)
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)


# =============================================================================
# Configuration
# =============================================================================

# Update this path to point at your local dataset.
# Expected layout: <MAIN_DATA_PATH>/{entrainement,validation,test}/<class>/*.jpg
MAIN_DATA_PATH = "/kaggle/input/datasets/majdabouhou/donnee/donnees/"
MODEL_SAVE_PATH = "/kaggle/working/Model.keras"

TRAIN_PATH = MAIN_DATA_PATH + "entrainement"
VAL_PATH = MAIN_DATA_PATH + "validation"

# Dataset sizes (one batch per epoch — see __next__() calls below)
TRAINING_BATCH_SIZE = 6160
VALIDATION_BATCH_SIZE = 1540

# Image config
IMAGE_SCALE = 224
IMAGE_CHANNELS = 3
IMAGE_COLOR_MODE = "rgb"
IMAGE_SHAPE = (IMAGE_SCALE, IMAGE_SCALE, IMAGE_CHANNELS)

# Training config
FIT_BATCH_SIZE = 32
FIT_EPOCHS = 100
NUM_CLASSES = 6
LEARNING_RATE = 0.0003
LABEL_SMOOTHING = 0.1


# =============================================================================
# GPU setup
# =============================================================================

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# =============================================================================
# Model
# =============================================================================

def feature_extraction(x):
    """Four-block convolutional feature extractor.

    Each block is two Conv2D(3x3) layers with BatchNorm + ReLU, followed by
    MaxPooling2D(2x2) and Dropout(0.3). Filter counts: 32, 64, 128, 256.
    """
    for filters in (32, 64, 128, 256):
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
    return x


def fully_connected(encoded):
    """Classifier head.

    GlobalAveragePooling2D is used instead of Flatten: Flatten would have
    produced a 50,176-dim vector, which led to immediate overfitting in early
    experiments. GAP keeps the head dimensions at 256 -> 128 -> 6 and trains
    much more stably.
    """
    x = GlobalAveragePooling2D()(encoded)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(NUM_CLASSES)(x)
    return Activation("softmax")(x)


def build_model():
    inputs = Input(shape=IMAGE_SHAPE)
    outputs = fully_connected(feature_extraction(inputs))
    model = Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# Data loading
# =============================================================================

def build_data_generators():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        zoom_range=0.2,
        shear_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_iter = train_gen.flow_from_directory(
        TRAIN_PATH,
        color_mode=IMAGE_COLOR_MODE,
        target_size=(IMAGE_SCALE, IMAGE_SCALE),
        batch_size=TRAINING_BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_iter = val_gen.flow_from_directory(
        VAL_PATH,
        color_mode=IMAGE_COLOR_MODE,
        target_size=(IMAGE_SCALE, IMAGE_SCALE),
        batch_size=VALIDATION_BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    print("Training class indices:  ", train_iter.class_indices)
    print("Validation class indices:", val_iter.class_indices)

    return train_iter, val_iter


# =============================================================================
# Training
# =============================================================================

def main():
    model = build_model()
    model.summary()

    train_iter, val_iter = build_data_generators()
    x_train, y_train = train_iter.__next__()
    x_val, y_val = val_iter.__next__()

    # Compensate for class imbalance in the training set
    # (e.g., 1600 tiger images vs 640 giraffe images)
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights_array = compute_class_weight(
        "balanced", classes=np.unique(y_train_labels), y=y_train_labels,
    )
    class_weight_dict = dict(enumerate(class_weights_array))

    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="auto",
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
    ]

    start = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=FIT_EPOCHS,
        batch_size=FIT_BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        shuffle=True,
    )
    duration = time.time() - start

    print(f"\n>>> Total training time: {duration:.2f}s ({duration / 60:.2f} min)")
    print(f">>> Min training loss:    {min(history.history['loss']):.4f}")
    print(f">>> Max training accuracy: {max(history.history['accuracy']):.4f}")

    # Plot accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"])
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"])
    plt.show()


if __name__ == "__main__":
    main()
