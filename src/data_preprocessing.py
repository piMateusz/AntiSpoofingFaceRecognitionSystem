import tensorflow as tf
import numpy as np


def preprocess_data(images, new_size):
    """
    1. Resize data to common image size
    2. Normalize values to 0 - 1
    :param data:
    :param new_size:
    :return:
    """
    # TODO consider changing images to tensors
    #  images: 4 - D Tensor of shape[batch, height, width, channels] or 3 - D Tensor of shape[height, width, channels].

    # Resized images will be distorted if their original aspect ratio is not the same as size.
    # To avoid distortions see tf.image.resize_with_pad. !
    resized = np.zeros_like(images)
    for counter, image in enumerate(images):
        resized[counter] = tf.image.resize(image, [new_size, new_size], method=tf.image.ResizeMethod.BILINEAR,
                                           preserve_aspect_ratio=False, antialias=False, name=None)

    preprocessed_data = resized / 255

    return preprocessed_data
