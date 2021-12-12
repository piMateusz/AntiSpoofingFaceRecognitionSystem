import logging
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint

from image_utils import get_all_images
from plot_utils import plot_img_and_cropped, plot_model_evaluation
from model import make_model_cnn
from constants import DATASET_PATH, TRAIN_IMAGE_JSON_PATH, TEST_IMAGE_JSON_PATH, \
    TF_RECORD_TRAIN_DIR_PATH, TF_RECORD_TEST_DIR_PATH, COMMON_IMAGE_SIZE, CHECKPOINT_FILE_PATH, BATCH_SIZE

import tensorflow as tf

if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    # get image datasets
    train_dataset = get_all_images(DATASET_PATH, TRAIN_IMAGE_JSON_PATH, TF_RECORD_TRAIN_DIR_PATH, COMMON_IMAGE_SIZE)
    test_dataset = get_all_images(DATASET_PATH, TEST_IMAGE_JSON_PATH, TF_RECORD_TEST_DIR_PATH, COMMON_IMAGE_SIZE)
    end = time.time()
    logging.info(f" Total loading images execution time: {end - start}")
    # plot one example image (default and cropped)
    # plot_img_and_cropped(all_cropped_images[0], all_default_images[0])

    start = time.time()
    cnn_model = make_model_cnn()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH,
                         monitor='val_accuracy',
                         verbose=1,
                         save_weights_only=False,
                         save_best_only=True,
                         mode='max')

    # history = model.fit(train_dataset, epochs=10, callbacks=[es, mc], validation_data=test_dataset)
    history = cnn_model.fit(train_dataset, epochs=10, callbacks=[mc], validation_data=test_dataset)

    # Recreate the exact same model, including its weights and the optimizer
    # cnn_model = tf.keras.models.load_model(CHECKPOINT_FILE_PATH)

    # Show the model architecture
    # cnn_model.summary()

    plot_model_evaluation(history)

    # test_loss, test_acc = model.evaluate(test_dataset)
    test_loss, test_acc = cnn_model.evaluate(test_dataset)

    print(f"accuracy: {test_acc}")
    predictions = cnn_model.predict(test_dataset, verbose=1)
    print(f"predictions: {predictions > 0.5}")

    end = time.time()
    logging.info(f" Total training time: {end - start}")
