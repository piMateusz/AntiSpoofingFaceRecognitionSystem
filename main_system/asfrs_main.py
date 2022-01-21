import cv2
import dlib

import tensorflow as tf

from retinaface import RetinaFace

from anty_spoofing.constants import COMMON_IMAGE_SIZE, BATCH_SIZE, CHECKPOINT_RESNET_FILE_PATH
from anty_spoofing.data_preprocessing import preprocess_data
from face_recognition_system.src.face_recognition_system import classify_one_image, get_encoded_faces
from constants import ANTI_SPOOFING_ACCESS_DENIED_MSG, ANTI_SPOOFING_ACCESS_GRANTED_MSG, \
    FACE_RECOGNITION_ACCESS_GRANTED_MSG, FACE_RECOGNITION_ACCESS_DENIED_MSG, DATABASE_IMAGE_FOLDER_PATH


def face_recognition_subsystem(image, database_faces_encoded):
    face_recognition_result = classify_one_image(database_faces=database_faces_encoded, input_image=image)
    user_msg = FACE_RECOGNITION_ACCESS_DENIED_MSG if face_recognition_result == "Unknown" \
        else FACE_RECOGNITION_ACCESS_GRANTED_MSG.format(face_recognition_result).replace("_", " ")
    
    return user_msg


def anti_spoofing_subsystem(image, coords, model):
    left, down, right, up = coords
    cropped_image = image[up: down, left: right]
    cropped_image = tf.convert_to_tensor(cropped_image)
    preprocessed_image = preprocess_data(cropped_image, COMMON_IMAGE_SIZE)
    dataset = tf.data.Dataset.from_tensors(preprocessed_image)
    dataset = dataset.batch(BATCH_SIZE)
    # make prediction
    prediction = model.predict(dataset)
    # print(f"prediction from image: {prediction}")
    anti_spoofing_result = prediction > 0.5

    user_msg = ANTI_SPOOFING_ACCESS_DENIED_MSG if anti_spoofing_result == 1 else ANTI_SPOOFING_ACCESS_GRANTED_MSG
    
    return user_msg


def draw_ui(user_msg, image, coords):
    # drawing constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (40, 40)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    # if face was found - draw a box around face
    if coords:
        left, down, right, up = coords
        cv2.rectangle(image, (left, up), (right, down), (0, 0, 255), 2)

    # display message to user
    if "\n" in user_msg:
        user_msg, user_msg_second_line = user_msg.split("\n")[0], user_msg.split("\n")[1]
        cv2.putText(image, user_msg_second_line, (40, 90), font, font_scale, font_color, thickness, line_type)
    cv2.putText(image, user_msg, bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

    # Display the resulting frame
    cv2.imshow('frame', image)
    cv2.waitKey(1)


def main():
    """ System initialization """
    # define a video capture object
    vid = cv2.VideoCapture(0)

    # load anti spoofing model
    model = tf.keras.models.load_model(CHECKPOINT_RESNET_FILE_PATH)

    # get encoded faces from database (image folder)
    database_faces_encoded = get_encoded_faces(DATABASE_IMAGE_FOLDER_PATH)

    # initialize face detector
    face_detector = dlib.get_frontal_face_detector()

    """ Main system loop """
    while True:
        # Capture the video frame
        ret, image = vid.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # crop image using face_recognition
        detected_faces = face_detector(image_rgb, 1)
        coords = ()
        for i, face_rect in enumerate(detected_faces):
            coords = face_rect.left(), face_rect.bottom(), face_rect.right(), face_rect.top()
            break       # program takes into consideration only first found face

        # crop image using RetinaFace
        # resp = RetinaFace.detect_faces(image_rgb)
        # coords = resp["facial_area"]

        if not coords:
            user_msg = "Nie znaleziono twarzy"
        else:
            """ Anti spoofing subsystem """
            user_msg = anti_spoofing_subsystem(image_rgb, coords, model)
            """ Face recognition subsystem """
            if user_msg == ANTI_SPOOFING_ACCESS_GRANTED_MSG:
                user_msg = face_recognition_subsystem(image_rgb, database_faces_encoded)
        
        draw_ui(user_msg, image, coords)
        
        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
