import os
import tensorflow as tf

# choose testing case
TESTING_CASE = "intra_test"
# TESTING_CASE = "protocol1"
# TESTING_CASE = "protocol2

# files paths
DATASET_PATH = os.path.join("..", "data", "celebA_spoof", "CelebA_Spoof")
LOCAL_TRAIN_IMAGE_JSON_PATH = os.path.join("metas", TESTING_CASE, "train_label.json")
LOCAL_TEST_IMAGE_JSON_PATH = os.path.join("metas", TESTING_CASE, "test_label.json")
TRAIN_IMAGE_JSON_PATH = os.path.join(DATASET_PATH, LOCAL_TRAIN_IMAGE_JSON_PATH)
TEST_IMAGE_JSON_PATH = os.path.join(DATASET_PATH, LOCAL_TEST_IMAGE_JSON_PATH)

CHECKPOINT_DIR_PATH = os.path.join("..", "checkpoints")
CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_DIR_PATH, "best_model_cnn.hdf5")
# tf record file paths
TF_RECORD_TRAIN_DIR_PATH = os.path.join(DATASET_PATH, "Data", TESTING_CASE, "train")
TF_RECORD_TEST_DIR_PATH = os.path.join(DATASET_PATH, "Data", TESTING_CASE, "test")

# image constants
COMMON_IMAGE_SIZE = 224

# learning constants
BATCH_SIZE = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE
