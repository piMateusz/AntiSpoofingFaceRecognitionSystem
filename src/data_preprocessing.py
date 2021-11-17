import tensorflow as tf


def preprocess_data(image, new_size):
    """
    1. Resize data to common image size
    2. Normalize values to 0 - 1
    :param image:
    :param new_size:
    :return:
    """
    # Resized images will be distorted if their original aspect ratio is not the same as size.
    # To avoid distortions see tf.image.resize_with_pad. !

    resized = tf.image.resize(image, [new_size, new_size], method=tf.image.ResizeMethod.BILINEAR,
                              preserve_aspect_ratio=False, antialias=False, name=None)

    preprocessed_data = resized / 255

    return preprocessed_data
