
from stock_reader import BasicReader


class MatrixReader(BasicReader):

    def __init__(self, path, index_file, sequence_length):
        super().__init__(path, index_file, sequence_length)

    def get_feature_from_df(self, df):
        raw_feature = df[['change_percent', 'high_percent',
                          'low_percent', 'index_change']].values
        return raw_feature

