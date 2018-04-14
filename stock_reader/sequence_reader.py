from stock_reader import BasicReader
import pandas as pd
import numpy as np


class SequenceReader(BasicReader):

    def __init__(self, path):
        super().__init__(path)
        self.fields = ["p_change"]
        self.seq_len = 20

    # def combine_result(self, res, new):
    #     return np.vstack((res, new))

    # def is_empty(self, new):
    #     if new is None:
    #         return True
    #     elif new.size == 0:
    #         return True
    #     else:
    #         return False

    def load_file(self, file_name):
        df = pd.read_csv(file_name)[self.fields]
        # num = self.seq_len + 1
        if len(df) <= self.seq_len * 2:
            return None, None

        features = []
        targets = []

        for index in range(len(df) - self.seq_len - 1):
            array = df[index: index + self.seq_len + 1].values
            feature = array[0:self.seq_len].flatten()
            target = array[self.seq_len]

            features.append(feature)
            targets.append(target)

        return targets, features

