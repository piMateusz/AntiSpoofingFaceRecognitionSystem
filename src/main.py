import logging
import time

from image_utils import get_all_images
from plot_utils import plot_img_and_cropped, plot_model_evaluation
from model import make_model
from constants import DATASET_PATH, IMAGE_JSON_PATH, TF_RECORD_TRAIN_FILE_PATH, TF_RECORD_TEST_FILE_PATH, COMMON_IMAGE_SIZE


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    # get image datasets
    train_dataset = get_all_images(DATASET_PATH, IMAGE_JSON_PATH, TF_RECORD_TRAIN_FILE_PATH, COMMON_IMAGE_SIZE)
    test_dataset = get_all_images(DATASET_PATH, IMAGE_JSON_PATH, TF_RECORD_TEST_FILE_PATH, COMMON_IMAGE_SIZE)

    # plot one example image (default and cropped)
    # plot_img_and_cropped(all_cropped_images[0], all_default_images[0])

    # model = make_model()
    #
    # history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    #
    # plot_model_evaluation(history)
    #
    # test_loss, test_acc = model.evaluate(test_dataset)
    #
    # print(f"accuracy: {test_acc}")

    end = time.time()
    logging.info(f" Total execution time: {end - start}")


