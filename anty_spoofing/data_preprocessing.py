import tensorflow as tf
from keras.applications import imagenet_utils


def preprocess_data(image: tf.float32, new_size: int)->tf.float32:
    """
    1. Resize data to common image size
    2. Normalize values to 0 - 1
    :param image: image to resize
    :param new_size: new size of image
    :return: normalized image
    """
    resized = tf.image.resize(image, [new_size, new_size], method=tf.image.ResizeMethod.BILINEAR,
                              preserve_aspect_ratio=False, antialias=False, name=None)

    preprocessed_data = resized / 255

    return preprocessed_data
