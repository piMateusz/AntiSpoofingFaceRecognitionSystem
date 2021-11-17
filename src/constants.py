import os
import tensorflow as tf

# files paths
DATASET_PATH = os.path.join("..", "data", "celebA_spoof", "CelebA_Spoof")
LOCAL_IMAGE_JSON_PATH = os.path.join("metas", "intra_test", "test_label.json")
IMAGE_JSON_PATH = os.path.join(DATASET_PATH, LOCAL_IMAGE_JSON_PATH)
# TODO change this name later
TF_RECORD_TRAIN_FILE_PATH = os.path.join(DATASET_PATH, "Data", 'images.tfrecords')
TF_RECORD_TEST_FILE_PATH = os.path.join(DATASET_PATH, "Data", 'images_test.tfrecords')

# image constants
COMMON_IMAGE_SIZE = 224


# learning constants
BATCH_SIZE = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
