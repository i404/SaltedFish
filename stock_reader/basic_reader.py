import os
import pandas as pd
import numpy as np

from stock_reader.reader import Reader


def load_preprocessed_data(feature_file, target_file):
    features = pd.read_csv(feature_file, header=None).values
    targets = pd.read_csv(target_file, header=None).values
    return features, targets


class BasicReader(Reader):

    def __init__(self, path, feature_range=20, data_min_length=50):
        self.path = path
        self.feature_range = feature_range
        self.df_min_length = data_min_length

    def get_feature_from_df(self, df):
        raise NotImplementedError("get_feature_from_df")

    @staticmethod
    def get_target_from_df(df):
        # return df["p_change"].values[1]
        arr = df["open"].values
        return (arr[0] - arr[1]) / arr[1]

    @staticmethod
    def format_target(target):
        return [1 if x > 0 else 0 for x in target]

    def load_raw_data(self):

        file_lst = []

        train_targets = []
        train_features = []

        validation_targets = []
        validation_features = []

        def add_target_feature_to_res(df, index, targets, features):
            # target = (open(t) - open(t-1)) / open(t-1)
            # need two days to get target
            df_target = df[index: index + 2]
            df_feature = df[index + 2: index + self.feature_range + 2]

            target = self.get_target_from_df(df_target)
            feature = self.get_feature_from_df(df_feature)

            targets.append(target)
            features.append(feature)

        for file_name in os.listdir(self.path):
            file_name = os.path.join(self.path, file_name)
            df = pd.read_csv(file_name)
            if not len(df) < self.df_min_length:
                file_lst.append(file_name)
                add_target_feature_to_res(df, 0, validation_targets, validation_features)
                for index in range(1, len(df) - self.feature_range - 2):
                    add_target_feature_to_res(df, index, train_targets, train_features)

        train_targets = self.format_target(train_targets)
        validation_targets = self.format_target(validation_targets)

        # return np.array(train_targets), np.array(train_features)
        res = [np.array(lst)
               for lst in [train_targets, train_features,
                           validation_targets, validation_features]]

        return {"stock_ids": file_lst,
                "train_targets": res[0],
                "train_features": res[1],
                "validation_targets": res[2],
                "validation_features": res[3]}

    # def load_raw_data_p_change(self):
    #     train_targets = []
    #     train_features = []
    #
    #     validation_targets = []
    #     validation_features = []
    #
    #     def add_target_feature_to_res(df, index, targets, features):
    #         # target = (open(t) - open(t-1)) / open(t-1)
    #         # need two days to get target
    #         df_target = df[index: index + 1]
    #         df_feature = df[index + 1: index + self.feature_range + 1]
    #
    #         target = df_target["p_change"].values[0]
    #         feature = self.get_feature_from_df(df_feature)
    #
    #         targets.append(target)
    #         features.append(feature)
    #
    #     for file_name in os.listdir(self.path):
    #         file_name = os.path.join(self.path, file_name)
    #         df = pd.read_csv(file_name)
    #         if not len(df) < self.df_min_length:
    #             add_target_feature_to_res(df, 0, validation_targets, validation_features)
    #             for index in range(1, len(df) - self.feature_range - 1):
    #                 add_target_feature_to_res(df, index, train_targets, train_features)
    #
    #     train_targets = self.format_target(train_targets)
    #     validation_targets = self.format_target(validation_targets)
    #
    #     # return np.array(train_targets), np.array(train_features)
    #     return [np.array(lst)
    #             for lst in [train_targets, train_features,
    #                         validation_targets, validation_features]]
