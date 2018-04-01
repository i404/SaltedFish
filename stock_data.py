import os
import numpy as np
import pandas as pd
import keras.backend as K

import config


def load_all_seq(path, fields, seq_len):
    load_fun = load_seq(fields, seq_len)
    return load_all(path, load_fun)


def load_seq(fields, seq_len):
    def tmp(file_name):
        df = pd.read_csv(file_name)[fields]
        data_set = []
        num = seq_len + 1
        if len(df) <= num * 2:
            return None
        for index in range(len(df) - num):
            data_set.append(df[index: index + num].values)
        return np.array(data_set)

    return tmp


def np_vstack_combine(a, b):
    def is_none(x):
        if x is None or x.size is 0:
            return True
        else:
            return False

    if is_none(a) and is_none(b):
        return None
    elif is_none(a):
        return b
    elif is_none(b):
        return a
    else:
        return np.vstack((a, b))


def load_all(path, load_fun, combine_fun=np_vstack_combine):
    res = None
    for file_name in os.listdir(path):
        file_name = os.path.join(path, file_name)
        tmp_arr = load_fun(file_name)
        if res is None and tmp_arr is None:
            res = None
        elif res is None:
            res = tmp_arr
        elif tmp_arr is None:
            res = res
        else:
            res = combine_fun(res, tmp_arr)
    return res


def load_data_for_cnn(filename, length=20):
    df = pd.read_csv(filename)
    df = df.drop(columns=["date"])
    if len(df) < length * 2:
        return None

    # ncol = df.shape[1]
    # nrow = length

    targets = []
    features = []
    for index in range(0, len(df) - length - 1):
        target = 1 if df["p_change"][index] > 0 else 0
        targets.append(target)

        feature = df[index + 1: index + 1 + length].values
        features.append(feature)
    return targets, features


def combine_for_cnn(res: (list, list), a: (list, list)) -> (list, list):
    r_targets, r_features = res
    a_targets, a_features = a
    return r_targets + a_targets, r_features + a_features


def load_all_for_cnn(path):
    targets, features = load_all(path, load_data_for_cnn, combine_for_cnn)
    point_num = len(targets)
    img_rows = features[0].shape[0]
    img_cols = features[0].shape[1]
    features = np.array(features)
    if K.image_data_format() == 'channels_first':
        features = features.reshape(point_num, 1, img_rows, img_cols)
        config.input_shape = (1, img_rows, img_cols)
    else:
        features = features.reshape(point_num, img_rows, img_cols, 1)
        config.input_shape = (img_rows, img_cols, 1)

    return targets, features


if __name__ == "__main__":
    t_target, t_feature = load_all_for_cnn("data_test")
    # print(features[:10])
    print(1)
