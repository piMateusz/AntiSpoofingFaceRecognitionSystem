import os
import tensorflow as tf

# choose testing case
TESTING_CASE = "intra_test"
# TESTING_CASE = "protocol1"
# TESTING_CASE = os.path.join("protocol2", "test_on_middle_quality_device")

# files paths
DATASET_PATH = os.path.join("..", "data", "celebA_spoof", "CelebA_Spoof")
LOCAL_TRAIN_IMAGE_JSON_PATH = os.path.join("metas", TESTING_CASE, "train_label.json")
LOCAL_TEST_IMAGE_JSON_PATH = os.path.join("metas", TESTING_CASE, "test_label.json")
TRAIN_IMAGE_JSON_PATH = os.path.join(DATASET_PATH, LOCAL_TRAIN_IMAGE_JSON_PATH)
TEST_IMAGE_JSON_PATH = os.path.join(DATASET_PATH, LOCAL_TEST_IMAGE_JSON_PATH)

CHECKPOINT_DIR_PATH = os.path.join("..", "checkpoints")
CHECKPOINT_CNN_FILE_PATH = os.path.join(CHECKPOINT_DIR_PATH, "best_model_cnn.hdf5")
# res_net50_best_model
CHECKPOINT_RESNET_FILE_PATH = os.path.join(CHECKPOINT_DIR_PATH, "best_model_resnet50.hdf5")
# lastly changed res_net50
CHECKPOINT_LAST_RESNET_FILE_PATH = os.path.join(CHECKPOINT_DIR_PATH, "last_model_resnet50.hdf5")

# tf record file paths - small dataset for res_net50
TF_RECORD_TRAIN_DIR_PATH = os.path.join(DATASET_PATH, "Data", TESTING_CASE, "train_maly_zbior")
TF_RECORD_TEST_DIR_PATH = os.path.join(DATASET_PATH, "Data", TESTING_CASE, "test_maly_zbior")

# tf record file paths - whole dataset
# TF_RECORD_TRAIN_DIR_PATH = os.path.join(DATASET_PATH, "Data", TESTING_CASE, "train")
# TF_RECORD_TEST_DIR_PATH = os.path.join(DATASET_PATH, "Data", TESTING_CASE, "test")

# image constants
COMMON_IMAGE_SIZE = 224
INPUT_SHAPE = (COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE, 3)

# learning constants
BATCH_SIZE = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
