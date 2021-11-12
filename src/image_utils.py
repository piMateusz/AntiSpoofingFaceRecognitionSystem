import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import json

from typing import Tuple, List, Dict
from constants import DATASET_PATH, IMAGE_JSON_PATH, ALL_CROPPED_IMAGES_NPY_PATH,\
                        ALL_DEFAULT_IMAGES_NPY_PATH, ALL_LABELS_NPY_PATH


def read_image(image_local_path: str) -> Tuple[np.ndarray, str]:
    image_path = os.path.join(DATASET_PATH, image_local_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cut image extension and append bbox prefix
    img_bbox_path = image_path[:-4] + "_BB.txt"

    assert os.path.exists(img_bbox_path), f"path {img_bbox_path} does not exist"

    return img, img_bbox_path


def crop_image(image: np.ndarray, image_bbox_path: str) -> np.ndarray:
    real_h, real_w, _ = image.shape
    cropped_img = image[:]

    with open(image_bbox_path, 'r') as file:
        bbox = file.readline()

        try:
            x, y, w, h, score = bbox.strip().split(" ")

        except Exception as e:
            logging.error(f" Error reading image bbox {image_bbox_path}: {e}")

        try:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            # get real bounding box
            x1 = int(x * (real_w / 224))
            y1 = int(y * (real_h / 224))
            w1 = int(w * (real_w / 224))
            h1 = int(h * (real_h / 224))

            # crop face from image
            x_1 = 0 if x1 < 0 else x1
            y_1 = 0 if y1 < 0 else y1
            x_2 = real_w if x_1 + w1 > real_w else x1 + w1
            y_2 = real_h if y_1 + h1 > real_h else y1 + h1

            cropped_img = image[y_1:y_2, x_1:x_2, :]

        except Exception as e:
            logging.error(f" Error in cropping face from image bbox {image_bbox_path}: {e}")

    return cropped_img


def get_all_images(all_image_dict: Dict[str, List[int]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """

    :param all_image_dict: dictionary with all images in format:
            key: path of image
            value: label of image; [0:40]: face attribute labels, [40]: spoof type label,
                                    [41]: illumination label, [42]: Environment label [43]: live/spoof label
    :return: generator object which contains tuples with image, cropped image and live/spoof label
    """
    logging.info(" Getting all images started")
    images_number = len(all_image_dict)

    for counter, image_local_path in enumerate(all_image_dict):
        # TODO remove brake when split array task is done
        if counter % 1000 == 0 and counter:
            logging.info(f" Loaded {counter} images")
        if counter == 10:
            break
        try:
            img, img_bbox_path = read_image(image_local_path)
            cropped_img = crop_image(img, img_bbox_path)
            live_spoof_label = all_image_dict[image_local_path][-1]
            yield img, cropped_img, live_spoof_label

        except Exception as e:
            logging.error(f" Error in getting all images for image {image_local_path}: {e}")
            images_number -= 1

    logging.info(f" Successfully loaded [{images_number}/{len(all_image_dict)}] images")


def read_json_file(all_image_json_path: str):
    try:
        with open(all_image_json_path) as file:
            all_image_list = json.load(file)
            logging.info(f" Successfully read json file: {all_image_json_path}")

    except Exception as e:
        logging.error(f" Could not read json file {all_image_json_path}: {e}")

    return all_image_list


def get_images_as_np_array(cropped=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param cropped: if True gets cropped images else default images
    :return: Tuple of all images and labels read from npy files if they exist otherwise from get_all_images() function
    """
    npy_file_path = ALL_CROPPED_IMAGES_NPY_PATH if cropped is True else ALL_DEFAULT_IMAGES_NPY_PATH
    all_images = None
    labels = None

    if os.path.exists(npy_file_path):

        try:
            all_images = np.load(npy_file_path, allow_pickle=True)
            logging.info(f" Loaded images from file: {npy_file_path}")

        except Exception as e:
            logging.error(f" Error in loading images from file {npy_file_path}: {e}")

        try:
            labels = np.load(ALL_LABELS_NPY_PATH, allow_pickle=True)
            logging.info(f" Loaded labels from file: {ALL_LABELS_NPY_PATH}")

        except Exception as e:
            logging.error(f" Error in loading labels from file {npy_file_path}: {e}")

    else:
        all_images_dict = read_json_file(IMAGE_JSON_PATH)
        images_iter = get_all_images(all_images_dict)

        all_default_images = []
        all_cropped_images = []
        labels = []

        for counter, value in enumerate(images_iter):
            img, cropped_img, label = value[0], value[1], value[2]
            all_default_images.append(img)
            all_cropped_images.append(cropped_img)
            labels.append(label)

        all_cropped_images = np.array(all_cropped_images)
        np.save(ALL_CROPPED_IMAGES_NPY_PATH, all_cropped_images)
        logging.info(f" Successfully saved cropped images at {ALL_CROPPED_IMAGES_NPY_PATH}")

        all_default_images = np.array(all_default_images)
        np.save(ALL_DEFAULT_IMAGES_NPY_PATH, all_default_images)
        logging.info(f" Successfully saved default images at {ALL_DEFAULT_IMAGES_NPY_PATH}")

        labels = np.array(labels)
        np.save(ALL_LABELS_NPY_PATH, labels)
        logging.info(f" Successfully saved labels at {ALL_LABELS_NPY_PATH}")

        all_images = all_cropped_images if cropped is True else all_default_images

    return all_images, labels
