import os

ANTI_SPOOFING_ACCESS_DENIED_MSG = "Odmowa dostepu.\nWykryto probe podszywania sie."
ANTI_SPOOFING_ACCESS_GRANTED_MSG = "Przynano dostep"
FACE_RECOGNITION_ACCESS_GRANTED_MSG = "Przyznano dostep.\nWitaj {} !"
FACE_RECOGNITION_ACCESS_DENIED_MSG = "Odmowa dostepu.\nBrak uprawnien."

DATA_FOLDER_PATH = os.path.join("..", "face_recognition_system", "data")
DATABASE_IMAGE_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "images_database")
