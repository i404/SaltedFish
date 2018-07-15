from stock_reader import BasicReader
import pandas as pd
import numpy as np


class SequenceReader(BasicReader):

    def __init__(self, path, index_file):
        super().__init__(path, index_file)

    def get_feature_from_df(self, df):
        return df["change_percent"].values
