import os
import cv2
import logging
import face_recognition
import numpy as np

from typing import Dict, Tuple

from constants import DATABASE_IMAGE_FOLDER_PATH, UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH, \
    UNKNOWN_LIVE_FALSE_IMAGE_FOLDER_PATH, UNKNOWN_SPOOF_IMAGE_FOLDER_PATH


def get_encoded_faces(image_folder_path: str, type_="live", label=True) -> Dict[str, Tuple[np.ndarray, str, bool]]:
    """
    encodes all faces from database (temporary solution - image folder)
    :param image_folder_path: path to folder with images
    :param type_: live/spoof image
    :param label: face recognition label - decide whether face should be recognized or not
    :return: dict of {name: image encoded, type, label}
    """

    encoded = {}

    for root, dirs, files in os.walk(image_folder_path):
        for img_file in files:
            face = face_recognition.load_image_file(os.path.join(root, img_file))
            try:
                encoding = face_recognition.face_encodings(face)[0]
                encoded[img_file.split(".")[0]] = (encoding, type_, label)
            except IndexError as e:
                logging.error(f"Could not get face encodings for image {img_file} with size {face.shape}. Error msg: {e}")
    return encoded


def classify_face(db_img_path: str, live_img_true_path: str, live_img_false_path: str, spoof_img_path: str):
    """
    recognize faces from files in given paths
    :param db_img_path: path to database images folder
    :param live_img_true_path: path to live images with True labels
    :param live_img_false_path: path to live images with False labels
    :param spoof_img_path: path to spoof images
    :return: list of dicts of {name: type, label, recognition result}
    """
    # get encoded faces from database (image folder)
    database_faces = get_encoded_faces(db_img_path)
    db_faces_encoded = [encoding for encoding, _, _ in database_faces.values()]
    db_faces_names = list(database_faces.keys())

    # get encoded unknown live true faces
    unknown_live_true_faces = get_encoded_faces(live_img_true_path, type_="live", label=True)

    # get encoded unknown live false faces
    unknown_live_false_faces = get_encoded_faces(live_img_false_path, type_="live", label=False)

    # get encoded unknown spoof faces
    unknown_spoof_faces = get_encoded_faces(spoof_img_path, type_="spoof", label=False)

    recognition_results = []
    unknown_faces_encoded = [unknown_live_true_faces, unknown_live_false_faces, unknown_spoof_faces]

    for faces_encoded_dict in unknown_faces_encoded:
        results = []
        for image_name in faces_encoded_dict:
            unknown_face_encoding, type, label = faces_encoded_dict[image_name]
            # TODO choose tolerance
            # face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)
            matches = face_recognition.compare_faces(db_faces_encoded, unknown_face_encoding)
            result = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(db_faces_encoded, unknown_face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                result = db_faces_names[best_match_index]

            result_dict = {image_name: (type, label, result)}
            results.append(result_dict)

        recognition_results.append(results)

    return recognition_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recognition_results_ = classify_face(DATABASE_IMAGE_FOLDER_PATH, UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH,
                                         UNKNOWN_LIVE_FALSE_IMAGE_FOLDER_PATH, UNKNOWN_SPOOF_IMAGE_FOLDER_PATH)
    for recognition_dict in recognition_results_:
        print(recognition_dict)


