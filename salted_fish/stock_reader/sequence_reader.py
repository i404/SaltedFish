from stock_reader import BasicReader


class SequenceReader(BasicReader):

    def __init__(self, path, index_file, sequence_length):
        super().__init__(path, index_file, sequence_length)

    def get_feature_from_df(self, df):
        return df["change_percent"].values
