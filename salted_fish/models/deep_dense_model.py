from keras import Sequential
from keras.layers import Dropout, Dense

from models import BasicModel
from stock_reader import SequenceReader
from util import bias_binary_crossentropy


class DeepDenseModel(BasicModel):

    def _create_reader(self):
        return SequenceReader(
            self.data_path, self.index_file, self.sequence_length)

    def _reshape_input(self, features):
        return features

    def __init__(self, input_shape=(64,), epochs=500):
        # self.input_dim = 64
        self.loss = bias_binary_crossentropy
        # self._epochs = 500
        super().__init__(input_shape=input_shape, epochs=epochs)

    def _create(self):
        model = Sequential()
        model.add(Dense(128, input_shape=self.input_shape, activation="relu"))
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
