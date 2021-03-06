import os
import dlib
import cv2
import face_recognition

import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import exposure, io
from typing import List

from constants import UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH, IMAGES_FOLDER_PATH, DATABASE_IMAGE_FOLDER_PATH

image_name = "303536.jpg"
image_path = os.path.join(UNKNOWN_LIVE_TRUE_IMAGE_FOLDER_PATH, image_name)

out_face_localization_file_name = "lokalizacja_twarzy.jpg"
out_face_localization_file_path = os.path.join(IMAGES_FOLDER_PATH, out_face_localization_file_name)

out_hog_file_name = "hog.jpg"
out_hog_file_path = os.path.join(IMAGES_FOLDER_PATH, out_hog_file_name)

out_face_landmarks_file_name = "face_landmarks.jpg"
out_face_landmarks_file_path = os.path.join(IMAGES_FOLDER_PATH, out_face_landmarks_file_name)

database_image_path = os.path.join(DATABASE_IMAGE_FOLDER_PATH, "celeb_1.jpg")
out_face_before_and_after_normalization_file_name = "face_before_and_after_normalization.jpg"
out_face_before_and_after_normalization_file_path = os.path.join(IMAGES_FOLDER_PATH,
                                                                 out_face_before_and_after_normalization_file_name)
out_face_landmarks_normalized_file_name = "face_landmarks_normalized.jpg"
out_face_landmarks_normalized_file_path = os.path.join(IMAGES_FOLDER_PATH, out_face_landmarks_normalized_file_name)

out_encodings_file_name = "encodings.jpg"
out_encodings_file_path = os.path.join(IMAGES_FOLDER_PATH, out_encodings_file_name)

out_face_normalized_and_encodings_file_name = "face_normalized_and_encodings.jpg"
out_face_normalized_and_encodings_file_path = os.path.join(IMAGES_FOLDER_PATH,
                                                           out_face_normalized_and_encodings_file_name)

out_pie_chart_live_faces_file_name = "pie_chart_live.png"
out_pie_chart_live_faces_file_path = os.path.join(IMAGES_FOLDER_PATH, out_pie_chart_live_faces_file_name)

out_pie_chart_spoof_faces_file_name = "pie_chart_spoof.png"
out_pie_chart_spoof_faces_file_path = os.path.join(IMAGES_FOLDER_PATH, out_pie_chart_spoof_faces_file_name)


def plot_face_bbox(input_image_path, output_image_path):
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()

    img = cv2.imread(input_image_path)

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(img, 1)

    print("I found {} faces in the file {}".format(len(detected_faces), input_image_path))
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i + 1, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))
        # Draw a box around each face we found
        cv2.rectangle(img, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)

    # save image
    cv2.imwrite(output_image_path, img)


def plot_img_and_hog(input_image_path, output_image_path):
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex="all", sharey="all")

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Obraz wej??ciowy')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram zorientowanych gradient??w')
    # plt.show()
    plt.savefig(output_image_path)


def plot_face_landmarks(input_image_path):
    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "shape_predictor_68_face_landmarks.dat"

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    win = dlib.image_window()

    # Load the image
    image = io.imread(input_image_path)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), input_image_path))

    # Show the desktop window with the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))
        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)
        # Draw the face landmarks on the screen.
        win.add_overlay(pose_landmarks)

    dlib.hit_enter_to_continue()


def plot_face_before_and_after_normalization(input_image_path, output_image_path, normalized_image_path):
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()

    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_normalized = cv2.imread(normalized_image_path)
    image_normalized = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB)

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(img, 1)
    cropped_image = []
    print("I found {} faces in the file {}".format(len(detected_faces), input_image_path))
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i + 1, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))
        cropped_image = img[face_rect.top(): face_rect.bottom(), face_rect.left(): face_rect.right()]
        # resize image
        cropped_image = cv2.resize(cropped_image, (image_normalized.shape[1], image_normalized.shape[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex="all", sharey="all")

    ax1.axis('off')
    ax1.imshow(cropped_image, cmap=plt.cm.gray)
    ax1.set_title('Nieznormalizowana twarz')

    ax2.axis('off')
    ax2.imshow(image_normalized, cmap=plt.cm.gray)
    ax2.set_title('Znormalizowana twarz')
    # plt.show()
    plt.savefig(output_image_path)


def create_encodings_image(input_image_path):
    face = face_recognition.load_image_file(input_image_path)
    encodings = face_recognition.face_encodings(face)[0]
    np.set_printoptions(linewidth=60, precision=3)
    print(f"encodings: \n{encodings}")


def plot_face_normalized_and_encodings(normalized_img_path, output_encodings_path, output_image_path):
    normalized_img = cv2.imread(normalized_img_path)
    normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)

    encodings_img = cv2.imread(output_encodings_path)
    encodings_img = cv2.cvtColor(encodings_img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(normalized_img, cmap=plt.cm.gray)
    ax1.set_title('Znormalizowana twarz')

    ax2.axis('off')
    ax2.imshow(encodings_img, cmap=plt.cm.gray)
    ax2.set_title('Cechy twarzy')
    # plt.show()
    plt.savefig(output_image_path)


def plot_pie_chart(values: List[int], labels: List[str], output_image_path):
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(output_image_path)
    # plt.show()


if __name__ == "__main__":
    # uncomment wanted functions

    # plot_face_bbox(image_path, out_face_localization_file_path)
    # plot_img_and_hog(image_path, out_hog_file_path)
    # plot_face_landmarks(image_path)
    # plot_face_before_and_after_normalization(database_image_path, out_face_before_and_after_normalization_file_path,
    #                                          out_face_landmarks_normalized_file_path)
    # create_encodings_image(database_image_path)
    # plot_face_normalized_and_encodings(out_face_landmarks_normalized_file_path, out_encodings_file_path,
    #                                    out_face_normalized_and_encodings_file_path)
    plot_pie_chart([116, 2], ["Rozpoznane twarze", "Nierozpoznane twarze"], out_pie_chart_live_faces_file_path)
    plot_pie_chart([528, 24], ["Rozpoznane twarze", "Nierozpoznane twarze"], out_pie_chart_spoof_faces_file_path)
