import os
import pandas as pd
import numpy as np


def load_preprocessed_data(feature_file, target_file):
    features = pd.read_csv(feature_file, header=None).values
    targets = pd.read_csv(target_file, header=None).values
    return features, targets


class BasicReader(object):

    def __init__(self, path):
        self.path = path

    def load_file(self, file_name):
        raise NotImplementedError("load_file")

    @staticmethod
    def format_target(target):
        return [1 if x > 0 else 0 for x in target]

    # def combine_result(self, res, new):
    #     raise NotImplementedError("combine_result")

    # def is_empty(self, new):
    #     raise NotImplementedError("is_empty")

    def load_raw_data(self):
        res_targets = []
        res_features = []
        for file_name in os.listdir(self.path):
            file_name = os.path.join(self.path, file_name)
            a_targets, a_features = self.load_file(file_name)
            if a_targets is not None and a_features is not None:
                res_targets += a_targets
                res_features += a_features

        res_targets = self.format_target(res_targets)

        return np.array(res_targets), np.array(res_features)

