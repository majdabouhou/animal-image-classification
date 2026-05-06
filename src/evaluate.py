"""
Wildlife Image Classification — Evaluation script
==================================================

Loads a trained model and evaluates it on the test set:
  - prints test loss and accuracy
  - plots the confusion matrix
  - displays one misclassified image per (true, predicted) class combination

Author: Majda Bouhou
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


# =============================================================================
# Configuration
# =============================================================================

# Update these to match your local setup.
MAIN_DATA_PATH = "/kaggle/input/datasets/majdabouhou/donnee/donnees/"
MODEL_PATH = "/kaggle/working/Model.keras"

TEST_PATH = MAIN_DATA_PATH + "test"

NUMBER_IMAGES_PER_CLASS = 200          # 200 per class, balanced test set
NUM_CLASSES = 6
IMAGE_SCALE = 224
IMAGE_COLOR_MODE = "rgb"

# Class names — must match the alphabetical folder ordering used by Keras
CLASS_NAMES = ["elephant", "girafe", "leopard", "rhino", "tigre", "zebre"]


# =============================================================================
# GPU setup
# =============================================================================

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# =============================================================================
# Helpers
# =============================================================================

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_misclassification_grid(cm, y_true, predicted_classes, filenames, test_path,
                                 class_names, num_classes):
    """Show one misclassified image per (true_class, predicted_class) cell.

    Diagonal cells display the count of correctly classified images instead.
    """
    fig, axes = plt.subplots(num_classes, num_classes, figsize=(18, 18))

    # Row labels (true) on the left, column labels (predicted) on top
    for i in range(num_classes):
        axes[i, 0].set_ylabel(
            class_names[i], fontsize=10, rotation=0, labelpad=60, va="center",
        )
        axes[0, i].set_title(class_names[i], fontsize=10)

    incorrect_idx = np.where(predicted_classes != y_true)[0]

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            ax = axes[true_class][pred_class]
            ax.set_xticks([])
            ax.set_yticks([])

            if true_class == pred_class:
                ax.text(
                    0.5, 0.5, f"{cm[true_class][pred_class]}",
                    ha="center", va="center", fontsize=12, color="green",
                    transform=ax.transAxes,
                )
                ax.set_facecolor("#f0f0f0")
                continue

            found = False
            for idx in incorrect_idx:
                if y_true[idx] == true_class and predicted_classes[idx] == pred_class:
                    img_path = test_path + "/" + filenames[idx]
                    ax.imshow(Image.open(img_path))
                    found = True
                    break

            if not found:
                ax.text(
                    0.5, 0.5, "0", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes,
                )
                ax.set_facecolor("#f0f0f0")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    # Load model
    classifier = load_model(MODEL_PATH)

    # Test data loader (one image at a time, no shuffle, so y_true ordering
    # matches the alphabetical folder layout)
    test_data_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_iter = test_data_generator.flow_from_directory(
        TEST_PATH,
        target_size=(IMAGE_SCALE, IMAGE_SCALE),
        class_mode="categorical",
        shuffle=False,
        batch_size=1,
        color_mode=IMAGE_COLOR_MODE,
    )

    # Ground truth — built directly from the alphabetical class ordering
    y_true = np.repeat(np.arange(NUM_CLASSES), NUMBER_IMAGES_PER_CLASS)

    # Evaluation
    test_eval = classifier.evaluate(test_iter, verbose=1)
    print(f">> Test loss:     {test_eval[0]:.4f}")
    print(f">> Test accuracy: {test_eval[1]:.4f}")

    # Predictions
    predicted_probs = classifier.predict(test_iter, verbose=1)
    predicted_classes = np.argmax(predicted_probs, axis=1)

    correct = np.sum(predicted_classes == y_true)
    incorrect = np.sum(predicted_classes != y_true)
    print(f"> {correct} correctly classified")
    print(f"> {incorrect} misclassified")

    # Confusion matrix
    cm = confusion_matrix(y_true, predicted_classes)
    plot_confusion_matrix(cm, CLASS_NAMES)

    # Misclassification grid
    plot_misclassification_grid(
        cm, y_true, predicted_classes, test_iter.filenames, TEST_PATH,
        CLASS_NAMES, NUM_CLASSES,
    )


if __name__ == "__main__":
    main()
