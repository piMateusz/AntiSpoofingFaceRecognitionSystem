import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import json

from typing import Tuple

DATASET_PATH = os.path.join("..", "data", "celebA_spoof", "CelebA_Spoof")
LOCAL_IMAGE_LIST_PATH = os.path.join("metas", "intra_test", "train_label.json")
IMAGE_LIST_PATH = os.path.join(DATASET_PATH, LOCAL_IMAGE_LIST_PATH)


def read_image(image_local_path: str) -> Tuple[np.ndarray, str]:
    image_path = os.path.join(DATASET_PATH, image_local_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cut image extension and append bbox prefix
    img_bbox_path = image_path[:-4] + "_BB.txt"
    assert os.path.exists(img_bbox_path), f"path {img_bbox_path} does not exist"
    return img, img_bbox_path


def crop_image(image: np.ndarray, image_bbox_path: str) -> np.ndarray:
    real_h, real_w = image.shape
    with open(image_bbox_path, 'r') as file:
        bbox = file.readline()
        try:
            x, y, w, h, score = bbox.strip().split(" ")
        except Exception as e:
            logging.info(f"Error reading image bbox {image_bbox_path}: {e}")
        try:
            # get real bounding box
            x1 = int(x * (real_w / 224))
            y1 = int(y * (real_h / 224))
            w1 = int(w * (real_w / 224))
            h1 = int(h * (real_h / 224))

            # crop face from image
            x_1 = 0 if x1 < 0 else x1
            y_1 = 0 if y1 < 0 else y1
            x_2 = real_w if x_1 + w > real_w else x1 + w
            y_2 = real_h if y_1 + h > real_h else y1 + h

            cropped_img = image[y_1:y_2, x_1:x_2, :]

        except Exception as e:
            logging.info(f"Error in cropping face from image bbox {image_bbox_path}: {e}")

    return cropped_img


def get_all_images(all_image_json_path: str) -> np.ndarray:
    with open(all_image_json_path) as file:
        all_image_list = json.load(file)
    for counter, image in enumerate(all_image_list):
        if counter == 0:
            print(image)


def plot_img_and_cropped(img, cropped):
    plt.imshow(img)
    plt.imshow(cropped)
    # cv2.imshow("full img", img)
    # cv2.waitKey(0)  # wait for a keyboard input
    # cv2.imshow("cropped img", cropped)
    # cv2.waitKey(0)  # wait for a keyboard input
