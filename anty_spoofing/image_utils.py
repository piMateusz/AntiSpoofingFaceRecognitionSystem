import os
import cv2
import logging
import json
import numpy as np
import tensorflow as tf

from typing import Tuple, List, Dict

from data_preprocessing import preprocess_data
from TFRecord_utils import save_data_to_tf_records, load_data_from_tf_records


def read_image(data_path: str, image_local_path: str) -> Tuple[np.ndarray, str]:
    """
    Read image from file
    :param data_path: path to CelebA_Spoof folder with data
    :param image_local_path: local path of image
    :return: tuple of image and path to image's bounding box coords
    """
    image_path = os.path.join(data_path, image_local_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cut image extension and append bbox prefix
    img_bbox_path = image_path[:-4] + "_BB.txt"

    assert os.path.exists(img_bbox_path), f"path {img_bbox_path} does not exist"

    return img, img_bbox_path


def read_json_file(all_image_json_path: str) -> Dict[str, List[int]]:
    """

    :param all_image_json_path: path to json file with train/test labels
    :return: dictionary in format:
            key: local path of image
            value: list of image's attribute's labels; [0:40]: face attribute labels, [40]: spoof type label,
                                    [41]: illumination label, [42]: Environment label [43]: live/spoof label
    """
    try:
        with open(all_image_json_path) as file:
            all_image_list = json.load(file)
            logging.info(f" Successfully read json file: {all_image_json_path}")

    except Exception as e:
        logging.error(f" Could not read json file {all_image_json_path}: {e}")

    return all_image_list


def preprocess_all_images(data_path: str, common_image_size: int, all_image_dict: Dict[str, List[int]]):
    """
    Reads all images, convert to tensors and preprocess them
    Pre-processing includes:
        1. Crop face from image
        2. Resize image
        3. Normalize image
    :param data_path: path to CelebA_Spoof folder with data
    :param common_image_size: all images will be resized to this size
    :param all_image_dict: dictionary in format:
            key: local path of image
            value: list of image's attribute's labels; [0:40]: face attribute labels, [40]: spoof type label,
                                    [41]: illumination label, [42]: Environment label [43]: live/spoof label
    :return: generator object which contains tuples with image, cropped image and live/spoof label
    """
    logging.info(" Getting all images started")
    images_number = len(all_image_dict)

    for counter, image_local_path in enumerate(all_image_dict):
        try:
            img, img_bbox_path = read_image(data_path, image_local_path)
            cropped_img = crop_image(img, img_bbox_path)
            # img = tf.convert_to_tensor(img)
            cropped_img = tf.convert_to_tensor(cropped_img)
            # preprocess cropped img
            cropped_img = preprocess_data(cropped_img, common_image_size)
            # get live/spoof attribute label
            live_spoof_label = all_image_dict[image_local_path][-1]
            yield cropped_img, live_spoof_label

        except Exception as e:
            logging.error(f" Error in getting all images for image {image_local_path}: {e}")
            images_number -= 1

    logging.info(f" Successfully loaded [{images_number}/{len(all_image_dict)}] images")


def crop_image(image: np.ndarray, image_bbox_path: str) -> np.ndarray:
    """
    Crop face from image
    :param image: image as np array
    :param image_bbox_path: path to image bounding box file
    :return: cropped image
    """
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


def get_all_images(data_path: str, json_path: str, tf_record_directory_path: str, common_image_size: int,
                   tf_record_size: int = 2000):
    """
    Get all images and labels
    If tf_record_file_path exists read images from TFRecord file
    else read images from files (.jpg, .png) and save to TFRecord file
    :param data_path: path to CelebA_Spoof folder with data
    :param json_path: path to json file with train/test labels
    :param tf_record_directory_path: path to TFRecords directory
    :param common_image_size: all images will be resized to this size
    :param tf_record_size: amount of images to save to one tf record
    :return: TFRecordDataset
    """

    if not os.path.exists(tf_record_directory_path):
        os.mkdir(tf_record_directory_path)
        all_images_dict = read_json_file(json_path)
        images_iter = preprocess_all_images(data_path, common_image_size, all_images_dict)

        all_cropped_images = []
        labels = []

        for counter, value in enumerate(images_iter):
            cropped_img, label = value[0], value[1]
            all_cropped_images.append(cropped_img)
            labels.append(label)
            if counter % tf_record_size == 0 and counter:
                all_cropped_images = tf.stack(all_cropped_images, axis=0, name='stack cropped images')
                tf_record_file_name = 'tfr_' + str(counter//tf_record_size) + '.tfrecords'
                tf_record_file_path = os.path.join(tf_record_directory_path, tf_record_file_name)
                logging.info(f" Saving {counter} images to: {tf_record_file_name}")
                save_data_to_tf_records(all_cropped_images, labels, tf_record_file_path)
                all_cropped_images = []

    tfr_file_pattern = os.path.join(tf_record_directory_path, "*.tfrecords")
    tfr_filenames = tf.io.gfile.glob(tfr_file_pattern)
    image_dataset = load_data_from_tf_records(tfr_filenames)

    return image_dataset
