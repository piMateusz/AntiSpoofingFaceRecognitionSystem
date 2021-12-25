import logging
import time
import os

import seaborn as sn
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix
from image_utils import get_all_images
from plot_utils import plot_model_accuracy, plot_model_loss, plot_sigmoid_function, plot_relu_function
from model import make_model_cnn
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from constants import DATASET_PATH, TRAIN_IMAGE_JSON_PATH, TEST_IMAGE_JSON_PATH, \
    TF_RECORD_TRAIN_DIR_PATH, TF_RECORD_TEST_DIR_PATH, COMMON_IMAGE_SIZE, CHECKPOINT_FILE_PATH, BATCH_SIZE, \
    CHECKPOINT_CNN_FILE_PATH


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    # get image datasets
    # train_dataset = get_all_images(DATASET_PATH, TRAIN_IMAGE_JSON_PATH, TF_RECORD_TRAIN_DIR_PATH, COMMON_IMAGE_SIZE)
    # test_dataset = get_all_images(DATASET_PATH, TEST_IMAGE_JSON_PATH, TF_RECORD_TEST_DIR_PATH, COMMON_IMAGE_SIZE)
    # val_dataset = get_all_images(DATASET_PATH, TEST_IMAGE_JSON_PATH, TF_RECORD_VALIDATION_DIR_PATH, COMMON_IMAGE_SIZE)
    plot_sigmoid_function()
    plot_relu_function()
    end = time.time()
    logging.info(f" Total loading images execution time: {end - start}")

    # cnn_model = tf.keras.models.load_model(CHECKPOINT_CNN_FILE_PATH)
    # cnn_model.summary()
    # plot_model(cnn_model, to_file="model_architecture.png", show_layer_names=False)

    # TODO ADD metrics !
    # 1. precision, recall, etc (Table)

    # start = time.time()
    #
    # cnn_model = make_model_cnn()
    #
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.001)
    # mc = ModelCheckpoint(filepath=CHECKPOINT_CNN_FILE_PATH,
    #                      monitor='val_accuracy',
    #                      verbose=1,
    #                      save_weights_only=False,
    #                      save_best_only=True,
    #                      mode='max')
    #
    # # history = model.fit(train_dataset, epochs=10, callbacks=[es, mc], validation_data=test_dataset)
    # history = cnn_model.fit(train_dataset, epochs=10, callbacks=[mc], validation_data=test_dataset)
    #
    # # Recreate the exact same model, including its weights and the optimizer
    # # cnn_model = tf.keras.models.load_model(CHECKPOINT_FILE_PATH)
    #
    # plot_model_accuracy(history)
    #
    # plot_model_loss(history)
    # # test_loss, test_acc = model.evaluate(test_dataset)
    # test_loss, test_acc = cnn_model.evaluate(test_dataset)
    #
    # print(f"accuracy: {test_acc}")
    # # predictions = model.predict(test_dataset, verbose=1)
    # print(f"predicting validation dataset")
    # predictions = cnn_model.predict(val_dataset, verbose=1)
    # pred_labels = predictions > 0.5
    #
    # print(f"pred labels shape: {pred_labels.shape}")
    #
    # # code for creating confusion matrix
    # print(f"creating conf matrix for val data")
    # test_labels = tf.constant([], dtype=tf.dtypes.int64)
    # for _, label_arr in val_dataset:
    #     test_labels = tf.concat([test_labels, label_arr], axis=0)
    #
    # print(f"test labels shape: {test_labels.shape}")
    # conf_matrix = confusion_matrix(test_labels, pred_labels)
    # print(f"conf matrix: {conf_matrix}")
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(conf_matrix, annot=False)
    # plt.show()
    #
    # end = time.time()
    # logging.info(f" Total training time: {end - start}")

    # code to get class distribution from tensorflow dataset (without batch size)

    # import numpy as np
    # vals = np.unique(np.fromiter(test_dataset.map(lambda x, y: y), float), return_counts=True)
    #
    # for val, count in zip(*vals):
    #     print(int(val), count)

    # code to plot image and label for debug purposes

    # for counter, (img, label) in enumerate(test_dataset):
    #     if counter < 5:
    #         print(f"batch no: {counter+1} label: {label}")
    #         plt.imshow(img)
    #         plt.axis('off')
    #         plt.title("Preprocessed image", fontsize=20)
    #         plt.show()

    """ ResNet50"""

    # Create a model and train it on the augmented image data
    # from tensorflow import keras
    # input_shape = (224, 224, 3)
    # inputs = keras.Input(shape=input_shape)
    # outputs = keras.applications.ResNet50(  # Add the rest of the model
    #     weights=None, input_shape=input_shape, classes=1)(inputs)
    # model = keras.Model(inputs, outputs)
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    # model.summary()
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    # mc = ModelCheckpoint(filepath=CHECKPOINT_RESNET_FILE_PATH,
    #                      monitor='val_accuracy',
    #                      verbose=1,
    #                      save_weights_only=False,
    #                      save_best_only=True,
    #                      mode='max')
    # history = model.fit(train_dataset, epochs=2, callbacks=[mc], validation_data=test_dataset)
