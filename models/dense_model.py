from keras import Sequential
from keras.layers import Dense, Dropout
from keras.losses import binary_crossentropy

from util import bias_binary_crossentropy
from models import BaseModel


class DenseModel(BaseModel):

    def __init__(self, input_shape=(20,), epochs=200):
        # self.input_dim = 20
        self.loss = binary_crossentropy
        # self._epochs = 200

        super().__init__(input_shape=input_shape, epochs=epochs)

    def _create(self):
        model = Sequential()
        model.add(Dense(40, input_shape=self.input_shape, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation="sigmoid"))

        # model.compile(loss="binary_crossentropy", optimizer="adam",
        model.compile(loss=self.loss, optimizer="adam",
                      metrics=["accuracy"])
        return model
