from stock_reader import BasicReader


class MatrixReaderWithIdAndStatus(BasicReader):

    def __init__(self, path, index_file, sequence_length):
        super().__init__(path, index_file, sequence_length)

    def get_feature_from_df(self, df):
        seq_feature = df[['change_percent', 'high_percent',
                          'low_percent', 'index_change']].values
        ids = df["stock_id"].values[0]
        date = df["date"].values[0]
        other_stock_status = self.total_stock_change_status[date]
        return seq_feature, ids, other_stock_status
