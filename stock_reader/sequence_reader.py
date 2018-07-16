from stock_reader import BasicReader
import pandas as pd
import numpy as np


class SequenceReader(BasicReader):

    def __init__(self, path, index_file, sequence_length):
        super().__init__(path, index_file, sequence_length)

    def get_feature_from_df(self, df):
        return df["p_change"].values
