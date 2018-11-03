import numpy as np
from keras import backend as K


def reshape_2d_feature_for_2d_cnn(raw_features):
    raw_features = np.array(raw_features)
    samples, rows, cols = raw_features.shape
    if K.image_data_format() == 'channels_first':
        feature = raw_features.reshape(samples, 1, rows, cols)
        shape = (1, rows, cols)
    else:
        feature = raw_features.reshape(samples, rows, cols, 1)
        shape = (rows, cols, 1)
    return shape, feature


def reshape_2d_feature_for_1d_cnn(raw_features):
    raw_features = np.array(raw_features)
    samples, rows, channels = raw_features.shape
    if K.image_data_format() == 'channels_first':
        feature = raw_features.reshape(samples, channels, rows)
        shape = (channels, rows)
    else:
        feature = raw_features.reshape(samples, rows, channels)
        shape = (rows, channels)
    return shape, feature


def reshape_1d_feature_for_1d_cnn(raw_features):
    raw_features = np.array(raw_features)
    samples, dim = raw_features.shape
    if K.image_data_format() == 'channels_first':
        feature = raw_features.reshape(samples, 1, dim)
        shape = (1, dim)
    else:
        feature = raw_features.reshape(samples, dim, 1)
        shape = (dim, 1)
    return shape, feature
