from keras import Sequential, optimizers
from keras.layers import LSTM, Dropout, Activation, Dense

from models import BasicModel
from stock_reader import SequenceReader


class LstmModel(BasicModel):

    def _create_reader(self):
        return SequenceReader(
            self.data_path, self.index_file, self.sequence_length)

    def __init__(self):
        super().__init__()
        self.timesteps = 20
        self.data_dim = 1
        self._epochs = 200

    def _reshape_input(self, features):
        return features

    def _create(self):
        model = Sequential()
        model.add(LSTM(input_shape=(self.timesteps, self.data_dim),
                       output_dim=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Activation("relu"))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(input_dim=100, output_dim=1))
        # model.add(Activation("linear"))
        model.add(Activation("sigmoid"))

        return model

