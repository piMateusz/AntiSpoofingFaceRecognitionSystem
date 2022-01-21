import os
import logging
import random
import face_recognition
import shutil

import numpy as np

from typing import Dict, Tuple, List


def get_encoded_faces(image_folder_path: str, type_="live", label=True) -> Dict[str, Tuple[str, np.ndarray, str, bool]]:
    """
    encodes all faces from database (temporary solution - image folder)
    :param image_folder_path: path to folder with images
    :param type_: live/spoof image
    :param label: face recognition label - decide whether face should be recognized or not
    :return: dict of {name: root name, image encoded, type, label}
    """

    encoded = {}

    for root, dirs, files in os.walk(image_folder_path):
        path = os.path.normpath(root)
        name = path.split(os.sep)[-2]
        for img_file in files:
            if "jpg" in img_file or "png" in img_file:
                face = face_recognition.load_image_file(os.path.join(root, img_file))
                try:
                    encoding = face_recognition.face_encodings(face)[0]
                    encoded[img_file.split(".")[0]] = (name, encoding, type_, label)
                except IndexError as e:
                    logging.error(f"Could not get face encodings for image {img_file} with size {face.shape}. Error msg: {e}")
    return encoded


def prepare_database_folder(db_img_path: str, person_list: List[str]):
    """
    copies one random image from every live person from person_list to db folder
    :param db_img_path: path to database images folder
    :param person_list: list with person's indexes
    """
    for live_person_folder in person_list:
        path = os.path.normpath(live_person_folder)
        name = path.split(os.sep)[-2]
        images_to_choose = [file_path for file_path in os.listdir(live_person_folder)
                            if "jpg" in file_path or "png" in file_path]
        random_live_image_file = random.choice(images_to_choose)
        random_live_image_path = os.path.join(live_person_folder, random_live_image_file)
        copy_path = os.path.join(db_img_path, name)
        os.mkdir(copy_path)
        copy_path = os.path.join(copy_path, "live")
        os.mkdir(copy_path)
        shutil.copy(random_live_image_path, copy_path)
        logging.info(f" Saved image from path {random_live_image_path} to database folder to with path {copy_path}")


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
    db_faces_encoded = []
    db_faces_names = []
    for name, encoding, _, _ in database_faces.values():
        db_faces_encoded.append(encoding)
        db_faces_names.append(name)

    unknown_faces_encoded = []
    for live_img_true_file_path in live_img_true_path:
        # get encoded unknown live true faces
        unknown_live_true_faces = get_encoded_faces(live_img_true_file_path, type_="live", label=True)
        unknown_faces_encoded.append(unknown_live_true_faces)

    for live_img_false_file_path in live_img_false_path:
        # get encoded unknown live false faces
        unknown_live_false_faces = get_encoded_faces(live_img_false_file_path, type_="live", label=False)
        unknown_faces_encoded.append(unknown_live_false_faces)

    for spoof_img_file_path in spoof_img_path:
        # get encoded unknown spoof faces
        unknown_spoof_faces = get_encoded_faces(spoof_img_file_path, type_="spoof", label=False)
        unknown_faces_encoded.append(unknown_spoof_faces)

    recognition_results = []

    for faces_encoded_dict in unknown_faces_encoded:
        results = []
        for image_name in faces_encoded_dict:
            # default tolerance is set to 0.6
            name, unknown_face_encoding, type, label = faces_encoded_dict[image_name]
            matches = face_recognition.compare_faces(db_faces_encoded, unknown_face_encoding)
            result = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(db_faces_encoded, unknown_face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                result = db_faces_names[best_match_index]

            result_dict = {image_name: (name, type, label, result)}
            results.append(result_dict)

        recognition_results.append(results)

    return recognition_results


def encode_one_image(image: np.ndarray):
    encoding = None
    try:
        encoding = face_recognition.face_encodings(image)[0]
    except IndexError as e:
        logging.error(f"Could not get face encodings for image. Error msg: {e}")

    return encoding


def classify_one_image(database_faces, input_image: np.ndarray):
    db_faces_encoded = [encoding for _, encoding, _, _ in database_faces.values()]
    db_faces_names = list(database_faces.keys())
    # get encoded input image
    input_image_encoded = encode_one_image(input_image)
    # default tolerance is set to 0.6
    matches = face_recognition.compare_faces(db_faces_encoded, input_image_encoded)
    result = "Unknown"

    # use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(db_faces_encoded, input_image_encoded)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        result = db_faces_names[best_match_index]

    return result


def prepare_face_recognition_results(recognition_results):
    all_spoof = all_live = all_live_db = true_live_db = false_spoof = false_live = all_false = 0

    for recognition_list in recognition_results:
        for recognition_dict_ in recognition_list:
            for key in recognition_dict_:
                # image is live
                if recognition_dict_[key][1] == "live":
                    all_live += 1
                    # is in database
                    if recognition_dict_[key][2] is True:
                        all_live_db += 1
                        # was recognised
                        if recognition_dict_[key][0] == recognition_dict_[key][3]:
                            true_live_db += 1
                    # is not in database
                    else:
                        all_false += 1
                        # was recognised
                        if recognition_dict_[key][0] == recognition_dict_[key][3]:
                            false_live += 1
                # image is spoof
                else:
                    all_spoof += 1
                    # was recognised
                    if recognition_dict_[key][0] == recognition_dict_[key][3]:
                        false_spoof += 1

    print(f"all_spoof: {all_spoof}")
    print(f"all_live: {all_live}")
    print(f"all_live_db: {all_live_db}")
    print(f"true_live_db: {true_live_db}")
    print(f"false_spoof: {false_spoof}")
    print(f"false_live: {false_live}")
    print(f"all_false: {all_false}")

    return all_spoof, all_live, all_live_db, true_live_db, false_spoof, false_live, all_false
