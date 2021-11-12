import logging
import time

from image_utils import get_images_as_np_array, plot_img_and_cropped
from constants import COMMON_IMAGE_SIZE
from data_preprocessing import preprocess_data

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
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # evaluate
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(f"accuracy: {test_acc}")

    end = time.time()
    logging.info(f" Total execution time: {end - start}")


