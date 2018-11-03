from keras import Sequential, optimizers
from keras.layers import LSTM, Dropout, Activation, Dense

from models import BasicModel


class LstmModel(BasicModel):

    def __init__(self):
        self.timesteps = 20
        self.data_dim = 1
        self._epochs = 200

        super().__init__()

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
        rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
        # sgd = optimizers.sgd(lr=0.0001, decay=1e-6,
        #                      momentum=0.9, nesterov=True)
        # model.compile(loss="mean_squared_logarithmic_error", optimizer=rms,
        # model.compile(loss="kullback_leibler_divergence", optimizer=rms,
        model.compile(loss="binary_crossentropy", optimizer=rms,
                      metrics=['accuracy'])
        return model

