import keras
from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, MaxPooling1D

from models import Model
from util import bias_mean_square_error


class Cnn1DModel(Model):

    def __init__(self, input_shape):
        self._epochs = 500
        # self.loss = keras.losses.mean_squared_error
        self.loss = bias_mean_square_error
        self.input_shape = input_shape
        super().__init__()

    def _create(self):
        model = Sequential()

        model.add(Convolution1D(filters=16, kernel_size=3, padding="same",
                                activation="relu",
                                input_shape=self.input_shape))
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=32, kernel_size=3, padding="same",
                                activation="relu"))
        # model.add(MaxPooling1D())
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=64, kernel_size=3, padding="same",
                                activation="relu"))
        model.add(Dropout(0.5))

        model.add(Convolution1D(filters=32, kernel_size=3, padding="same",
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
