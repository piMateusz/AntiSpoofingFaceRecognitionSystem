import matplotlib.pyplot as plt
import math
import os
import numpy as np


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
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


def plot_model_loss(history):
    plt.plot(history.history['loss'], label='Strata zbioru treningowego')
    plt.plot(history.history['val_loss'], label='Strata zbioru walidacyjnego')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość funkcji straty')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
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
