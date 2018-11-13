import os
import pandas as pd
import numpy as np
from datetime import datetime as dt

from sklearn.preprocessing import MinMaxScaler

from stock_reader import Reader
from util import memorize_df


def read_total_stock_change_status(index_file, stock_path):
    res_df = read_index_from_file(index_file)[["date", "index_change_flag"]]
    for cur_df, stock_code in read_dfs_from_path(stock_path):
        cur_df = cur_df[["date", "change_flag"]]
        cur_df.columns = ["date", stock_code]
        res_df = res_df.join(cur_df.set_index("date"), on="date")
        res_df = res_df.fillna(0.0)
    date_status_dict = res_df.set_index("date").T.to_dict('list')
    date_ind_dict = {}
    ind_status_dict = {}
    for ind, (date, status) in enumerate(date_status_dict.items()):
        date_ind_dict[date] = ind
        ind_status_dict[ind] = status
    return date_ind_dict, ind_status_dict


def read_dfs_from_path(file_path):
    stock_lst = os.listdir(file_path)
    for stock_id, stock_code in enumerate(sorted(stock_lst)):
        stock_file = os.path.join(file_path, stock_code)
        try:
            yield read_csv_from_file(stock_file, stock_id), stock_code
        except KeyError:
            print(f"read {stock_file} fails")


@memorize_df
def read_index_from_file(index_file):
    index_data = pd.read_csv(index_file)[['日期', '涨跌幅']]
    index_data.columns = ["date", "index_change"]

    change_flag = [1 if v > 0 else 0 for v in index_data["index_change"]]
    index_data["index_change_flag"] = np.array(change_flag)

    scaler = MinMaxScaler()
    index_data["index_change"] = scaler.fit_transform(
        index_data["index_change"].values.reshape(-1, 1))

    return index_data


@memorize_df
def read_csv_from_file(file_name, stock_id):

    df = pd.read_csv(file_name)
    open_value = df['open'].values
    close_value = df['close'].values
    high_value = df['high'].values
    low_value = df['low'].values
    df['stock_id'] = stock_id

    df['change_percent'] = change_to_percent(close_value, open_value)
    change_flag = [1 if v > 0.5 else 0 for v in df["change_percent"]]
    df['change_flag'] = np.array(change_flag)

    df['high_percent'] = change_to_percent(high_value, open_value)
    df['low_percent'] = change_to_percent(low_value, open_value)
    return df.iloc[::-1]


def change_to_percent(value, open_value):
    raw = ((value - open_value) / open_value) * 10.0
    return np.array([format_result(x) for x in raw])


def format_result(x):
    """
    original percent is supposed in [-1, 1]
    format percent to [0, 1]
    :param x: change percent of stock
    :return: format percent
    """
    if x > 1:
        xx = 1
    elif x < -1:
        xx = -1
    else:
        xx = x
    return (xx + 1) / 2


def compare_date(date_a, date_b, date_format="%Y-%m-%d"):
    a = dt.strptime(date_a, date_format)
    b = dt.strptime(date_b, date_format)
    return a == b


def train_or_test():
    return np.random.uniform() >= 0.3


def format_target(target):
    return [1 if x > 0 else 0 for x in target]


class BasicReader(Reader):

    def __init__(self, path, index_file, sequence_length=20):
        self.path = path
        self.index_file = index_file
        self.sequence_length = sequence_length
        self.unit_df_length = self.sequence_length + 2

        self.index_data = read_index_from_file(index_file)[
            ["date", "index_change"]]

        self.start_date = self.index_data.iloc[0]['date']
        self.index_length = self.index_data.shape[0]

        self.single_day_stock_change_status = None
        self.date_ind_dict = None

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
            while end_index(i) < df.shape[0] and \
                    end_index(j) < self.index_length:
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
        raw_target = (arr[0] - arr[1]) / arr[1]
        return 1 if raw_target > 0 else 0

    def add_target_feature_to_res(self, df, targets, features):
        df_target = df[0: 2]
        df_feature = df[2:]

        target = self.get_target_from_df(df_target)
        feature = self.get_feature_from_df(df_feature)

        targets.append(target)
        features.append(feature)

    def load_raw_data(self):
        stock_codes = []
        train_targets = []
        train_features = []
        test_targets = []
        test_features = []
        validation_targets = []
        validation_features = []

        for df, stock_code in read_dfs_from_path(self.path):
            if len(df) >= self.unit_df_length:
                stock_codes.append(stock_code)
                for sub_df in self.continuous_df(df):
                    if compare_date(sub_df.iloc[0]['date'], self.start_date):
                        self.add_target_feature_to_res(
                            sub_df, validation_targets, validation_features)
                    elif train_or_test():
                        self.add_target_feature_to_res(
                            sub_df, train_targets, train_features)
                    else:
                        self.add_target_feature_to_res(
                            sub_df, test_targets, test_features)

        return {"stock_codes": stock_codes,
                "train_targets": np.array(train_targets),
                "train_features": train_features,
                "test_targets": np.array(test_targets),
                "test_features": test_features,
                "validation_targets": np.array(validation_targets),
                "validation_features": validation_features}

