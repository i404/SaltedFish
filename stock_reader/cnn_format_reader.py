from stock_reader import Reader
import keras.backend as K


class CnnFormatReader(Reader):
    """
    it is a decorator, with format output feature to which cnn accepts
    (samples, dim) -> (samples, 1, dim) or (samples, dim, 1)
    (samples, row, col) -> (samples, 1, row, col) or (samples, row, col, 1)
    """

    def __init__(self, raw_reader):
        self.raw_reader = raw_reader

    def load_raw_data(self):
        res = self.raw_reader.load_raw_data()

        train_features = res.get("train_features")
        validation_features = res.get("validation_features")

        ndim = train_features.ndim

        if ndim is 2:
            shape, train_features = self.reshape_1d_feature(train_features)
            _, validation_features = self.reshape_1d_feature(validation_features)
        elif ndim is 3:
            shape, train_features = self.reshape_2d_feature(train_features)
            _, validation_features = self.reshape_2d_feature(validation_features)
        else:
            shape = None

        res["train_features"] = train_features
        res["validation_features"] = validation_features
        res["shape"] = shape

        return res

    @staticmethod
    def reshape_2d_feature(raw_features):
        samples, rows, cols = raw_features.shape
        if K.image_data_format() == 'channels_first':
            feature = raw_features.reshape(samples, 1, rows, cols)
            shape = (1, rows, cols)
        else:
            feature = raw_features.reshape(samples, rows, cols, 1)
            shape = (rows, cols, 1)
        return shape, feature

    @staticmethod
    def reshape_1d_feature(raw_features):
        samples, dim = raw_features.shape
        if K.image_data_format() == 'channels_first':
            feature = raw_features.reshape(samples, 1, dim)
            shape = (1, dim)
        else:
            feature = raw_features.reshape(samples, dim, 1)
            shape = (dim, 1)
        return shape, feature
