from stock_reader import Reader
import keras.backend as K


class CnnFormatReader(Reader):
    """
    it is a decorator, with format output feature to which cnn accepts
    (samples, dim) -> (samples, channels, dim) or (samples, dim, channels)
    (samples, dim, channels) -> (samples, channels, dim) or (samples, dim, channels)
    (samples, row, col) -> (samples, channels, row, col) or (samples, row, col, channels)
    """

    def __init__(self, raw_reader, cnn_dim=1):
        if cnn_dim is not 1 and cnn_dim is not 2:
            raise Exception(f"cnn_dim must be 1 or 2, not is {cnn_dim}")
        self.cnn_dim = cnn_dim
        self.raw_reader = raw_reader

    def load_raw_data(self):
        res = self.raw_reader.load_raw_data()

        train_features = res.get("train_features")
        validation_features = res.get("validation_features")

        ndim = train_features.ndim

        if ndim is 2:
            reshape_fun = self.reshape_1d_feature_for_1d_cnn
        elif ndim is 3:
            if self.cnn_dim == 2:
                reshape_fun = self.reshape_2d_feature_for_2d_cnn
            else:
                # self.cnn_dim == 1
                reshape_fun = self.reshape_2d_feature_for_1d_cnn
        else:
            raise Exception(f"input feature dim should be 2 or 3 instead of {ndim}")

        shape, train_features = reshape_fun(train_features)
        _, validation_features = reshape_fun(validation_features)

        res["train_features"] = train_features
        res["validation_features"] = validation_features
        res["shape"] = shape

        return res

    @staticmethod
    def reshape_2d_feature_for_2d_cnn(raw_features):
        samples, rows, cols = raw_features.shape
        if K.image_data_format() == 'channels_first':
            feature = raw_features.reshape(samples, 1, rows, cols)
            shape = (1, rows, cols)
        else:
            feature = raw_features.reshape(samples, rows, cols, 1)
            shape = (rows, cols, 1)
        return shape, feature

    @staticmethod
    def reshape_2d_feature_for_1d_cnn(raw_features):
        samples, rows, channels = raw_features.shape
        if K.image_data_format() == 'channels_first':
            feature = raw_features.reshape(samples, channels, rows)
            shape = (channels, rows)
        else:
            feature = raw_features.reshape(samples, rows, channels)
            shape = (rows, channels)
        return shape, feature

    @staticmethod
    def reshape_1d_feature_for_1d_cnn(raw_features):
        samples, dim = raw_features.shape
        if K.image_data_format() == 'channels_first':
            feature = raw_features.reshape(samples, 1, dim)
            shape = (1, dim)
        else:
            feature = raw_features.reshape(samples, dim, 1)
            shape = (dim, 1)
        return shape, feature
