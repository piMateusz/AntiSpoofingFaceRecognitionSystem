import os

# input files paths
DATASET_PATH = os.path.join("..", "data", "celebA_spoof", "CelebA_Spoof")
LOCAL_IMAGE_JSON_PATH = os.path.join("metas", "intra_test", "test_label.json")
IMAGE_JSON_PATH = os.path.join(DATASET_PATH, LOCAL_IMAGE_JSON_PATH)

# npy files paths
ALL_CROPPED_IMAGES_NPY_PATH = os.path.join(DATASET_PATH, "Data", "all_cropped_images.npy")
ALL_DEFAULT_IMAGES_NPY_PATH = os.path.join(DATASET_PATH, "Data", "all_default_images.npy")
ALL_LABELS_NPY_PATH = os.path.join(DATASET_PATH, "Data", "all_labels.npy")

COMMON_IMAGE_SIZE = 224
