import logging
import os

from constants import DATABASE_IMAGE_FOLDER_PATH, DATA_FOLDER_PATH, DATASET_PATH
from face_recognition_system import prepare_database_folder, classify_face, prepare_face_recognition_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # manually chosen people from CelebA-Spoof database for face recognition subsystem tests
    live_true_and_spoof_person_list = ["4930", "4931", "4943", "4966", "4973"]
    live_false_person_list = ["4978", "4989", "5033", "5035", "5051"]

    TEST_FILE_PATH = os.path.join(DATASET_PATH, "test")

    live_true_person_paths = [os.path.join(TEST_FILE_PATH, index, "live") for index in live_true_and_spoof_person_list]
    spoof_person_paths = [os.path.join(TEST_FILE_PATH, index, "spoof") for index in live_true_and_spoof_person_list]
    live_false_person_paths = [os.path.join(TEST_FILE_PATH, index, "live") for index in live_false_person_list]

    prepare_database_folder(DATABASE_IMAGE_FOLDER_PATH, live_true_person_paths)
    recognition_results_ = classify_face(DATABASE_IMAGE_FOLDER_PATH, live_true_person_paths,
                                         live_false_person_paths, spoof_person_paths)

    prepare_face_recognition_results(recognition_results_)
