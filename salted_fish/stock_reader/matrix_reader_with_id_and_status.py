from stock_reader import BasicReader
from .basic_reader import read_total_stock_change_status


class MatrixReaderWithIdAndStatus(BasicReader):

    def __init__(self, path, index_file, sequence_length):
        super().__init__(path, index_file, sequence_length)

    def load_raw_data(self):
        self.date_ind_dict, self.single_day_stock_change_status = \
            read_total_stock_change_status(self.index_file, self.path)
        return super().load_raw_data()

    def get_feature_from_df(self, df):
        seq_feature = df[['change_percent', 'high_percent',
                          'low_percent', 'index_change']].values
        ids = df["stock_id"].values[0]
        date = df["date"].values[0]
        date_ind = self.date_ind_dict[date]
        # return seq_feature, ids, date_ind
        return seq_feature, date_ind
