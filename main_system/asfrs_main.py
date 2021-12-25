import cv2
import dlib

import tensorflow as tf

from retinaface import RetinaFace

from anty_spoofing.constants import CHECKPOINT_CNN_FILE_PATH, COMMON_IMAGE_SIZE, BATCH_SIZE
from anty_spoofing.data_preprocessing import preprocess_data
from face_recognition_system.src.face_recognition_system import classify_one_image, get_encoded_faces
from constants import ANTI_SPOOFING_ACCESS_DENIED_MSG, ANTI_SPOOFING_ACCESS_GRANTED_MSG, \
    FACE_RECOGNITION_ACCESS_GRANTED_MSG, FACE_RECOGNITION_ACCESS_DENIED_MSG, DATABASE_IMAGE_FOLDER_PATH


def main():
    # TODO parametrize checkpoint file and common image size
    # TODO consider splitting this method to 2 - 3 functions (eg. face_recognition, anti_spoofing, drawing)
    # define a video capture object
    # vid = cv2.VideoCapture(0)


    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # set camera resolution
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # print camera resolution
    w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"resolution: {w}x{h}")


    # load anti spoofing model
    cnn_model = tf.keras.models.load_model(CHECKPOINT_CNN_FILE_PATH)

    # get encoded faces from database (image folder)
    database_faces_encoded = get_encoded_faces(DATABASE_IMAGE_FOLDER_PATH)

    while True:
        # Capture the video frame
        ret, image = vid.read()
        # import os
        # import time
        # image = cv2.imread(os.path.join("..", "face_recognition_system", "data", "unknown_spoof_images", "spoof.jpg"))
        # image = cv2.imread(os.path.join("..", "face_recognition_system", "data", "unknown_live_true_images", "live.jpg"))

        # cv2.imwrite("Mateusz_Pilecki.png", image)
        user_msg = ""
        # crop image
        # using face_recognition
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(image, 1)
        left = down = right = up = 0
        for i, face_rect in enumerate(detected_faces):
            left, down, right, up = face_rect.left(), face_rect.bottom(), face_rect.right(), face_rect.top()
            break       # this means program takes only first found face - consider changing
        if left == down == right == up == 0:
            user_msg = "Nie znaleziono twarzy"
        # using RetinaFace
        # resp = RetinaFace.detect_faces(image)
        # left, down, right, up = resp["facial_area"]     # "facial_area": [155, 81, 434, 443]

        if user_msg != "Nie znaleziono twarzy":
            """ Anti spoofing system """
            cropped_image = image[up: down, left: right]
            cropped_image = tf.convert_to_tensor(cropped_image)
            preprocessed_image = preprocess_data(cropped_image, COMMON_IMAGE_SIZE)
            dataset = tf.data.Dataset.from_tensors(preprocessed_image)
            dataset = dataset.batch(BATCH_SIZE)
            # make prediction
            prediction = cnn_model.predict(dataset)
            print(f"prediction: {prediction}")
            anti_spoofing_result = prediction < 0.9
            user_msg = ANTI_SPOOFING_ACCESS_DENIED_MSG if anti_spoofing_result == 1 else ANTI_SPOOFING_ACCESS_GRANTED_MSG

            """ Face recognition system """
            if user_msg == ANTI_SPOOFING_ACCESS_GRANTED_MSG:
                face_recognition_result = classify_one_image(database_faces=database_faces_encoded, input_image=image)
                user_msg = FACE_RECOGNITION_ACCESS_DENIED_MSG if face_recognition_result == "Unknown" \
                    else FACE_RECOGNITION_ACCESS_GRANTED_MSG.format(face_recognition_result).replace("_", " ")

        # Draw a box around face
        if user_msg != "Nie znaleziono twarzy":
            cv2.rectangle(image, (left, up), (right, down), (0, 0, 255), 2)

        # TODO display asfrs results
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0, 30)
        fontScale = 0.75
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2

        cv2.putText(image, user_msg, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        # Display the resulting frame
        cv2.imshow('frame', image)
        cv2.waitKey(1)

        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
