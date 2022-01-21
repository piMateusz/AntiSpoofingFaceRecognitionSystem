import logging
import time

import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint

from image_utils import get_all_images
from plot_utils import plot_model_accuracy, plot_model_loss, create_conf_matrix, plot_conf_matrix, \
    print_classification_report
from model import make_model_cnn, make_model_res_net_50
from constants import DATASET_PATH, TRAIN_IMAGE_JSON_PATH, TEST_IMAGE_JSON_PATH, CHECKPOINT_RESNET_FILE_PATH, \
    TF_RECORD_TRAIN_DIR_PATH, TF_RECORD_TEST_DIR_PATH, COMMON_IMAGE_SIZE, CHECKPOINT_LAST_RESNET_FILE_PATH, INPUT_SHAPE


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    """ GET TRAIN AND TEST DATASET"""
    start = time.time()

    train_dataset = get_all_images(DATASET_PATH, TRAIN_IMAGE_JSON_PATH, TF_RECORD_TRAIN_DIR_PATH, COMMON_IMAGE_SIZE)
    test_dataset = get_all_images(DATASET_PATH, TEST_IMAGE_JSON_PATH, TF_RECORD_TEST_DIR_PATH, COMMON_IMAGE_SIZE)

    end = time.time()
    logging.info(f" Total loading images execution time: {end - start}")

    """ LOAD MODEL FROM CHECKPOINT """
    start = time.time()
    # model = tf.keras.models.load_model(CHECKPOINT_RESNET_FILE_PATH)

    """ TRAIN MODEL """
    model = make_model_res_net_50(INPUT_SHAPE)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint(filepath=CHECKPOINT_LAST_RESNET_FILE_PATH, monitor='val_accuracy', verbose=1,
                         save_weights_only=False, save_best_only=True, mode='max')

    history = model.fit(train_dataset, epochs=20, callbacks=[mc, es], validation_data=test_dataset)

    logging.info(" Started model evaluation")
    predictions = model.predict(test_dataset, verbose=1)
    pred_labels = predictions > 0.5

    test_labels, conf_matrix = create_conf_matrix(test_dataset, pred_labels)
    print(f"conf matrix: \n{conf_matrix}")
    plot_conf_matrix(conf_matrix)

    # show classification report
    print_classification_report(test_labels, pred_labels)

    # plot and save model acc and loss figures
    plot_model_accuracy(history)
    plot_model_loss(history)

    end = time.time()
    logging.info(f" Total training time: {end - start}")
