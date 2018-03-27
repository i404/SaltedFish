import os
import numpy as np
import pandas as pd


def load_all_seq(path, fields, seq_len):
    load_fun = load_seq(fields, seq_len)
    return load_all(path, load_fun)


def load_seq(fields, seq_len):
    def tmp(file_name):
        df = pd.read_csv(file_name)[fields]
        data_set = []
        num = seq_len + 1
        if len(df) <= num:
            return None
        for index in range(len(df) - num):
            data_set.append(df[index: index+num].values)
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
        res = combine_fun(res, tmp_arr)
    return res
