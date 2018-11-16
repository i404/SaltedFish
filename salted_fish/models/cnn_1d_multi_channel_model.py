from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, BatchNormalization

from models import BasicModel
from reprocess import reshape_2d_feature_for_1d_cnn
from stock_reader import MatrixReader
from util import bias_mean_abs_error


class Cnn1DMultiChannelModel(BasicModel):

    def _create_reader(self):
        return MatrixReader(
            self.data_path, self.index_file, self.sequence_length)

    def __init__(self, *args, **kwargs):
        # self.loss = keras.losses.mean_squared_error
        # self.loss = keras.losses.mean_absolute_error
        # self.loss = bias_mean_square_error
        self.loss = bias_mean_abs_error
        self.kernel_size = 4
        super().__init__(*args, **kwargs)

    def _reshape_input(self, raw_features):
        shape, feature = reshape_2d_feature_for_1d_cnn(raw_features)
        self.input_shape = shape
        return feature

    def _create(self):

        model = Sequential()

        model.add(Convolution1D(filters=128, kernel_size=self.kernel_size,
                                padding="same", activation="relu",
                                # kernel_regularizer="l1",
                                input_shape=self.input_shape))
        model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(MaxPooling1D())

        model.add(Convolution1D(filters=64, kernel_size=self.kernel_size,
                                # kernel_regularizer="l1",
                                padding="same", activation="relu"))
        model.add(BatchNormalization())
        # model.add(MaxPooling1D())
        # model.add(Dropout(0.5))

        model.add(Convolution1D(filters=32, kernel_size=self.kernel_size,
                                # kernel_regularizer="l1",
                                padding="same", activation="relu"))
        model.add(BatchNormalization())
        # model.add(MaxPooling1D())
        # model.add(Dropout(0.5))

        model.add(Convolution1D(filters=16, kernel_size=self.kernel_size,
                                # kernel_regularizer="l1",
                                padding="same", activation="relu"))
        model.add(BatchNormalization())
        # model.add(MaxPooling1D())
        model.add(Dropout(0.2))

        model.add(Flatten())

        # model.add(Dense(1024, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(256, activation='relu'))

        # model.add(Dense(1, activation='linear'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss="binary_crossentropy",
            # loss=bias_binary_crossentropy,
            optimizer="adam",
            metrics=['accuracy'])
        return model
