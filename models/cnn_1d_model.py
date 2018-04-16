import keras
from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten

from models import Model


class Cnn1DModel(Model):

    def __init__(self, input_shape):
        self._epochs = 500
        self.input_shape = input_shape
        super().__init__()

    def _create(self):
        model = Sequential()

        model.add(Convolution1D(filters=8, kernel_size=3, padding="same",
                                activation="relu",
                                input_shape=self.input_shape))
        model.add(Dropout(0.2))

        model.add(Convolution1D(filters=8, kernel_size=3, padding="same",
                                activation="relu"))
        model.add(Dropout(0.2))

        model.add(Convolution1D(filters=8, kernel_size=3, padding="same",
                                activation="relu"))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))

        model.add(Dense(1, activation='softmax'))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer="adam",
                      metrics=['accuracy'])
        return model
