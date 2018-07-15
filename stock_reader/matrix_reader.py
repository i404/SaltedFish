from stock_reader import BasicReader
import pandas as pd
import keras.backend as K

from util import config


class MatrixReader(BasicReader):

    def __init__(self, path, index_file, cols=None):
        self.cols = cols
        super().__init__(path, index_file)

    def get_feature_from_df(self, df):
        raw_feature = df[['change_percent', 'high_percent', 'low_percent', 'index_change']]
        return raw_feature

