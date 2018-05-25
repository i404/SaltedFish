import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from stock_reader.reader import Reader


def load_preprocessed_data(feature_file, target_file):
    features = pd.read_csv(feature_file, header=None).values
    targets = pd.read_csv(target_file, header=None).values
    return features, targets


class BasicReader(Reader):

    def __init__(self, path, feature_range=20,
                 data_min_length=50):
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
        # return [1 if x > 0.001 else 0 for x in target]
        return [1 if x > 0 else 0 for x in target]

    def load_raw_data(self):

        stock_ids = []

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

        for stock_id in os.listdir(self.path):
            file_name = os.path.join(self.path, stock_id)
            df = pd.read_csv(file_name)
            if not len(df) < self.df_min_length:
                stock_ids.append(stock_id)
                add_target_feature_to_res(df, 0, validation_targets, validation_features)
                for index in range(1, len(df) - self.feature_range - 2):
                    add_target_feature_to_res(df, index, train_targets, train_features)

        train_targets = self.format_target(train_targets)
        validation_targets = self.format_target(validation_targets)

        return {"stock_ids": stock_ids,
                "train_targets": np.array(train_targets),
                "train_features": np.array(train_features),
                "validation_targets": np.array(validation_targets),
                "validation_features": np.array(validation_features)}

