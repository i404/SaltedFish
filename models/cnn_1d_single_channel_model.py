import keras
from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, MaxPooling1D, \
    BatchNormalization
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import SGD

from models import Model
from util import bias_mean_square_error, bias_mean_abs_error, bias_binary_crossentropy, tf_precision


class Cnn1DSingleChannelModel(Model):

    def __init__(self, epochs=500, batch_size=32, min_iter_num=20,
                 early_stop_epochs=None, verbose=1):
        # self.loss = keras.losses.mean_squared_error
        # self.loss = keras.losses.mean_absolute_error
        # self.loss = bias_mean_square_error
        self.loss = bias_mean_abs_error
        # self.loss = bias_binary_crossentropy
        # self.loss = "binary_crossentropy"
        self.kernel_size = 4
        super().__init__(epochs=epochs, batch_size=batch_size,
                         min_iter_num=min_iter_num,
                         early_stop_epochs=early_stop_epochs, verbose=verbose)

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

        model.compile(
            loss="binary_crossentropy",
            # loss=bias_binary_crossentropy,
            optimizer="adam",
            # optimizer="sgd",
            #  optimizer=opt,
            metrics=['accuracy'])
        return model
