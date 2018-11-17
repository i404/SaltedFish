from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, BatchNormalization

from models import BasicModel
from reprocess import reshape_1d_feature_for_1d_cnn
from stock_reader import SequenceReader
from util import bias_mean_abs_error


class Cnn1DSingleChannelModel(BasicModel):

    def _create_reader(self):
        return SequenceReader(
            self.data_path, self.index_file, self.sequence_length)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = bias_mean_abs_error
        self.kernel_size = 4

    def _reshape_input(self, raw_features):
        shape, feature = reshape_1d_feature_for_1d_cnn(raw_features)
        self.input_shape = shape
        return feature

    def _create(self):

        if self.input_shape is None:
            raise ValueError("input_shape is not set")

        model = Sequential()

        model.add(Convolution1D(filters=64, kernel_size=self.kernel_size,
                                padding="same", activation="relu",
                                # kernel_regularizer="l1",
                                input_shape=self.input_shape))
        model.add(BatchNormalization())
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
        model.add(Dropout(0.4))

        model.add(Flatten())

        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        # model.add(Dense(1, activation='linear'))
        model.add(Dense(1, activation='sigmoid'))

        # opt = SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)

        return model
