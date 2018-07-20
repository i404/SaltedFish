import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

from stock_reader.reader import Reader


def load_preprocessed_data(feature_file, target_file):
    features = pd.read_csv(feature_file, header=None).values
    targets = pd.read_csv(target_file, header=None).values
    return features, targets


def compare_date(date_a, date_b, date_format="%Y-%m-%d"):
    a = dt.strptime(date_a, date_format)
    b = dt.strptime(date_b, date_format)
    return a == b


class BasicReader(Reader):

    def __init__(self, path, index_file, sequence_length=20):
        self.path = path
        self.sequence_length = sequence_length
        self.unit_df_length = self.sequence_length + 2

        self.index_data = pd.read_csv(index_file)[['日期', '涨跌幅']]
        self.index_data.columns = ["date", "index_change"]
        self.start_date = self.index_data.iloc[0]['date']
        self.index_length = self.index_data.shape[0]

    def get_feature_from_df(self, df):
        raise NotImplementedError("get_feature_from_df")

    def continuous_df(self, df: pd.DataFrame):

        df = df.join(self.index_data.set_index("date"), on="date")

        if df.shape[0] == self.index_length:
            # this stock is full attendance
            for i in range(0, df.shape[0] - self.unit_df_length):
                yield df[i: i + self.unit_df_length]
        else:
            # not full
            i, j = 0, 0
            end_index = lambda x: x + self.unit_df_length
            while end_index(i) < df.shape[0] and end_index(j) < self.index_length:
                i_beg_date = df.iloc[i]['date']
                j_beg_date = self.index_data.iloc[j]['date']
                if compare_date(i_beg_date, j_beg_date):
                    i_end_date = df.iloc[end_index(i)]['date']
                    j_end_date = self.index_data.iloc[end_index(j)]['date']
                    if compare_date(i_end_date, j_end_date):
                        yield df[i: i + self.unit_df_length]
                    i += 1
                    j += 1
                else:
                    j += 1

    @staticmethod
    def get_target_from_df(df):
        # return df["p_change"].values[1]
        arr = df["open"].values
        return (arr[0] - arr[1]) / arr[1]

    @staticmethod
    def format_target(target):
        return [1 if x > 0 else 0 for x in target]

    @staticmethod
    def read_csv_from_file(file_name):
        df = pd.read_csv(file_name)
        open_value = df['open'].values
        close_value = df['close'].values
        high_value = df['high'].values
        low_value = df['low'].values
        change_percent = lambda x: ((x - open_value) / open_value) * 100.0
        df['change_percent'] = change_percent(close_value)
        df['high_percent'] = change_percent(high_value)
        df['low_percent'] = change_percent(low_value)
        return df.iloc[::-1]

    def load_raw_data(self):

        stock_ids = []

        train_targets = []
        train_features = []

        validation_targets = []
        validation_features = []

        def add_target_feature_to_res(df, targets, features):
            df_target = df[0: 2]
            df_feature = df[2:]

            target = self.get_target_from_df(df_target)
            feature = self.get_feature_from_df(df_feature)

            targets.append(target)
            features.append(feature)

        for stock_id in os.listdir(self.path):
            file_name = os.path.join(self.path, stock_id)
            # df = pd.read_csv(file_name)
            try:
                df = self.read_csv_from_file(file_name)
            except KeyError:
                print(f"read {file_name} fail")
                continue
            if not len(df) < self.unit_df_length:
                stock_ids.append(stock_id)
                for sub_df in self.continuous_df(df):
                    if compare_date(sub_df.iloc[0]['date'], self.start_date):
                        add_target_feature_to_res(sub_df, validation_targets, validation_features)
                    else:
                        add_target_feature_to_res(sub_df, train_targets, train_features)

        train_targets = self.format_target(train_targets)
        validation_targets = self.format_target(validation_targets)

        return {"stock_ids": stock_ids,
                "train_targets": np.array(train_targets),
                "train_features": np.array(train_features),
                "validation_targets": np.array(validation_targets),
                "validation_features": np.array(validation_features)}


if __name__ == "__main__":

    def test1():
        reader = BasicReader("data_test", "../total_index.csv")
        df = pd.read_csv("../data_test/_000001.csv")
        print(next(reader.continuous_df(df)))
        for sub_df in reader.continuous_df(df):
            print(sub_df['date'].values)


    def test2():
        print(BasicReader.read_csv_from_file("../data_test/000001.csv"))

    # test2()
