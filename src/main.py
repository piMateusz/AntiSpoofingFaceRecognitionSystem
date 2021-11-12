import logging
import time

from image_utils import get_images_as_np_array
from plot_utils import plot_img_and_cropped, plot_model_evaluation
from constants import COMMON_IMAGE_SIZE
from data_preprocessing import preprocess_data
from model import make_model

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    # get all images (default and cropped)
    all_cropped_images, labels = get_images_as_np_array(cropped=True)
    all_default_images, _ = get_images_as_np_array(cropped=False)
    # plot one example image (default and cropped)
    # plot_img_and_cropped(all_cropped_images[0], all_default_images[0])
    # print("len of cropped images: ", all_cropped_images.shape)
    # print("len of labels: ", labels.shape)
    # print(f"labels: {labels}")
    # print(f"all_cropped_images shape: {all_cropped_images[0].shape}")
    # preprocess data
    images = preprocess_data(all_cropped_images, COMMON_IMAGE_SIZE)
    # print(f"images shape: {images[0].shape}")
    # split data
    train_images = images[0:7]
    train_labels = labels[0:7]
    test_images = images[7:10]
    test_labels = labels[7:10]
    print(f"train_images shape: {train_images.shape}")
    print(f"train_images[0] shape: {train_images[0].shape}")
    #
    # print("train_images shape: ", train_images.shape)
    # print("test_images shape: ", test_images.shape)
    #
    model = make_model()

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plot_model_evaluation(history)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f"accuracy: {test_acc}")

    end = time.time()
    logging.info(f" Total execution time: {end - start}")


