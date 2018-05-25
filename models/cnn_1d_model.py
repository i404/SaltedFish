import keras
from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, MaxPooling1D

from models import Model
from util import bias_mean_square_error, bias_mean_abs_error


class Cnn1DModel(Model):

    def __init__(self, epochs=500, batch_size=32,
                 early_stop_epochs=None, verbose=1):
        # self.loss = keras.losses.mean_squared_error
        # self.loss = keras.losses.mean_absolute_error
        # self.loss = bias_mean_square_error
        self.loss = bias_mean_abs_error
        super().__init__(epochs=epochs, batch_size=batch_size,
                         early_stop_epochs=early_stop_epochs, verbose=verbose)

    def _create(self):

        if self.input_shape is None:
            raise ValueError("input_shape is not set")

        model = Sequential()

        model.add(Convolution1D(filters=64, kernel_size=3, padding="same",
                                activation="relu",
                                input_shape=self.input_shape))
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=32, kernel_size=3, padding="same",
                                activation="relu"))
        # model.add(MaxPooling1D())
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=32, kernel_size=3, padding="same",
                                activation="relu"))
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=16, kernel_size=3, padding="same",
                                activation="relu"))
        # model.add(MaxPooling1D())
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=16, kernel_size=3, padding="same",
                                activation="relu"))
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=8, kernel_size=3, padding="same",
                                activation="relu"))
        model.add(MaxPooling1D())
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))

        model.add(Dense(1, activation='linear'))

        model.compile(loss=self.loss,
                      optimizer="adam",
                      metrics=['accuracy'])
        return model
