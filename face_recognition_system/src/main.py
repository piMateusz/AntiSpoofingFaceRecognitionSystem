import logging

from constants import DATABASE_IMAGE_FOLDER_PATH, UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH, \
    UNKNOWN_LIVE_FALSE_IMAGE_FOLDER_PATH, UNKNOWN_SPOOF_IMAGE_FOLDER_PATH
from face_recognition_system import classify_face


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    recognition_results_ = classify_face(DATABASE_IMAGE_FOLDER_PATH, UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH,
                                         UNKNOWN_LIVE_FALSE_IMAGE_FOLDER_PATH, UNKNOWN_SPOOF_IMAGE_FOLDER_PATH)
    for recognition_dict in recognition_results_:
        print(recognition_dict)
