import os
import dlib
import cv2
import logging

from constants import UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH, IMAGES_FOLDER_PATH, DATABASE_IMAGE_FOLDER_PATH

image_name = "303536.jpg"
# image_path = os.path.join(UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH, image_name)
image_path = os.path.join(DATABASE_IMAGE_FOLDER_PATH, "celeb_1.jpg")

out_face_landmarks_normalized_file_name = "face_landmarks_normalized.jpg"
out_face_landmarks_normalized_file_path = os.path.join(IMAGES_FOLDER_PATH, out_face_landmarks_normalized_file_name)


def plot_face_landmarks_normalized(input_image_path, output_image_path):
    try:
        import openface
        # You can download the required pre-trained face detection model here:
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        predictor_model = "shape_predictor_68_face_landmarks.dat"
        # Create a HOG face detector using the built-in dlib class
        face_detector = dlib.get_frontal_face_detector()
        face_pose_predictor = dlib.shape_predictor(predictor_model)
        face_aligner = openface.AlignDlib(predictor_model)
        # Load the image
        image = cv2.imread(input_image_path)
        # Run the HOG face detector on the image data
        detected_faces = face_detector(image, 1)
        print("Found {} faces in the image file {}".format(len(detected_faces), input_image_path))
        # Loop through each face we found in the image
        for i, face_rect in enumerate(detected_faces):
            # Detected faces are returned as an object with the coordinates
            # of the top, left, right and bottom edges
            print(
                "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i + 1, face_rect.left(),
                                                                                   face_rect.top(),
                                                                                   face_rect.right(),
                                                                                   face_rect.bottom()))
            # Use openface to calculate and perform the face alignment
            alignedFace = face_aligner.align(534, image, face_rect,
                                             landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            # Save the aligned image to a file
            cv2.imwrite(output_image_path, alignedFace)

    except Exception as e:
        logging.error("Could not import openface library. Activate openface environment by typing in command line:\n"
                      "\'conda deactivate\' -> \'conda activate openface\'. Error msg: ", e)


if __name__ == "__main__":
    plot_face_landmarks_normalized(image_path, out_face_landmarks_normalized_file_path)