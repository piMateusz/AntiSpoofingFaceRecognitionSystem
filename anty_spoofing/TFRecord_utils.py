import tensorflow as tf
import logging
from functools import partial
from constants import BATCH_SIZE, AUTOTUNE

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()   # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(raw_image, label):
    """
    Create a dictionary with features that may be relevant.
    :param raw_image: serialized image (tensor)
    :param label: image label
    :return:
    """

    feature = {
        'raw_image': _bytes_feature(raw_image),
        'label': _int64_feature(label)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_data_to_tf_records(data, labels, tf_record_file_path):
    with tf.io.TFRecordWriter(tf_record_file_path) as writer:
        for image, label in zip(data, labels):
            # converting tensor to binary - strings
            image_serialized = tf.io.serialize_tensor(image)
            tf_example = image_example(image_serialized, label)
            writer.write(tf_example.SerializeToString())
    logging.info(f" Successfully saved preprocessed images with labels at {tf_record_file_path}")


# def load_data_from_tf_records(tf_record_file_path):
#     raw_image_dataset = tf.data.TFRecordDataset(tf_record_file_path)
#
#     # Create a dictionary describing the features.
#     image_feature_description = {
#         'raw_image': tf.io.FixedLenFeature([], tf.string),
#         'label': tf.io.FixedLenFeature([], tf.int64)
#     }
#
#     def _parse_image_function(example_proto):
#         # Parse the input tf.train.Example proto using the dictionary above.
#         features = tf.io.parse_single_example(example_proto, image_feature_description)
#         image, label = features['raw_image'], features['label']
#         image = tf.io.parse_tensor(features['raw_image'], out_type=tf.float32)
#         # image = tf.io.decode_raw(features['raw_image'], tf.uint8)
#         # label = tf.cast(features['label'], tf.int32)
#         # label = tf.one_hot(label, 10)
#
#         return image, label
#
#     # mapping parsing to raw image dataset
#     parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
#
#     logging.info(f" Loaded images dataset from file: {tf_record_file_path}")
#
#     return parsed_image_dataset


def read_tfrecord(example, labeled):
    image_feature_description = {
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, image_feature_description)
    image = tf.io.parse_tensor(example['raw_image'], out_type=tf.float32)
    if labeled:
        label = tf.cast(example["label"], tf.int64)
        return image, label
    return image


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled))
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def load_data_from_tf_records(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    # dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
