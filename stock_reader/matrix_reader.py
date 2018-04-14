from stock_reader import BasicReader
import pandas as pd


class MatrixReader(BasicReader):

    def __init__(self, path):
        super().__init__(path)
        self.length = 20

    def load_file(self, file_name):
        df = pd.read_csv(file_name)
        df = df.drop(columns=["date"])
        if len(df) < self.length * 2:
            return None, None

        targets = []
        features = []
        for index in range(0, len(df) - self.length - 1):
            target = 1 if df["p_change"][index] > 0 else 0
            targets.append(target)

            feature = df[index + 1: index + 1 + self.length].values
            features.append(feature)

        # if K.image_data_format() == 'channels_first':
        #     features = features.reshape(point_num, 1, img_rows, img_cols)
        #     config.input_shape = (1, img_rows, img_cols)
        # else:
        #     features = features.reshape(point_num, img_rows, img_cols, 1)
        #     config.input_shape = (img_rows, img_cols, 1)

        return targets, features

    # def combine_result(self, res, a):
    #     r_targets, r_features = res
    #     a_targets, a_features = a
    #     return r_targets + a_targets, r_features + a_features

    # def is_empty(self, a):
    #     if a is None:
    #         return True
    #     elif not len(a) == 2:
    #         return True
    #     elif len(a[0]) == 0:
    #         return True
    #     else:
    #         return False
