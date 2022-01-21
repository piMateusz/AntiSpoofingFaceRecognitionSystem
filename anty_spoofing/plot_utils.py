import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sn

import math
import os

from sklearn.metrics import confusion_matrix, classification_report


def plot_img_and_cropped(img, cropped):
    fig = plt.figure(figsize=(50, 50))

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Full image", fontsize=20)

    fig.add_subplot(1, 2, 2)
    plt.imshow(cropped)
    plt.axis('off')
    plt.title("Cropped image", fontsize=20)

    plt.show()


def plot_model_accuracy(history):
    plt.plot(history.history['accuracy'], label='Dokładność zbioru treningowego')
    plt.plot(history.history['val_accuracy'], label='Dokładność zbioru walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    # tick every second epoch on X axis
    plt.xticks(np.arange(1, len(history.history['accuracy']) + 1, 2))
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("..", "images", "model_accuracy.png"))
    plt.show()


def plot_model_loss(history):
    plt.plot(history.history['loss'], label='Strata zbioru treningowego')
    plt.plot(history.history['val_loss'], label='Strata zbioru walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość funkcji straty')
    # tick every second epoch on X axis
    plt.xticks(np.arange(1, len(history.history['loss']) + 1, 2))
    plt.ylim([0, max(history.history['val_loss'])])
    plt.legend(loc='upper right')
    plt.savefig(os.path.join("..", "images", "model_loss.png"))
    plt.show()


def sigmoid(x):
    a = []
    for item in x:
        a.append(1 / (1 + math.exp(-item)))
    return a


def reLU(x):
    a = []
    for item in x:
        if item < 0:
            a.append(0)
        else:
            a.append(item)
    return a


def plot_sigmoid_function():
    x = np.arange(-10., 10., 0.2)
    sig = sigmoid(x)
    plt.plot(x, sig)
    plt.savefig(os.path.join("..", "images", "sigmoid.png"))
    plt.show()


def plot_relu_function():
    x = np.arange(-10., 10., 0.2)
    relu = reLU(x)
    plt.plot(x, relu)
    plt.savefig(os.path.join("..", "images", "relu.png"))
    plt.show()


def create_conf_matrix(test_dataset, predictions):
    test_labels = tf.constant([], dtype=tf.dtypes.int64)
    for _, label_arr in test_dataset:
        test_labels = tf.concat([test_labels, label_arr], axis=0)
    conf_matrix = confusion_matrix(test_labels, predictions)
    return test_labels, conf_matrix


def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sn.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"fontsize": 18})
    plt.savefig(os.path.join("..", "images", "confusion_matrix.png"))
    plt.show()


def print_classification_report(test_labels, pred_labels):
    target_names = ['live', 'spoof']
    print(classification_report(test_labels, pred_labels, target_names=target_names))
