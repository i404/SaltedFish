from keras import Sequential
from keras.layers import Dropout, Dense

from util import bias_binary_crossentropy
from models import Model


class DeepDenseModel(Model):

    def __init__(self):
        self.input_dim = 64
        self.loss = bias_binary_crossentropy
        self._epochs = 500

        super().__init__()

    def _create(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(self.input_dim,), activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation="sigmoid"))

        # model.compile(loss="binary_crossentropy", optimizer="adam",
        model.compile(loss=self.loss, optimizer="adam",
                      metrics=["accuracy"])
        return model