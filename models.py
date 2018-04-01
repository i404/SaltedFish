import keras
from keras import Sequential, optimizers
from keras.layers import LSTM, Dropout, Activation, Dense, Conv2D, MaxPooling2D, Flatten

import config
from customized_loss import bias_loss


def create_dense_model(time_steps=20):
    """
    model with 3 dense layers
    :return: model
    """
    model = Sequential()
    model.add(Dense(40, input_shape=(time_steps,), activation="relu"))
    model.add(Dropout(0.2))

    # model.add(Dense(100, activation="relu"))
    # model.add(Dropout(0.2))

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    sgd = optimizers.sgd(lr=0.0001, decay=1e-6,
                         momentum=0.9, nesterov=True)
    # model.compile(loss="binary_crossentropy", optimizer="adam",
    model.compile(loss=bias_loss, optimizer="adam",
                  metrics=["accuracy"])
    return model


def create_lstm_model():
    data_dim = 1
    timesteps = 20
    model = Sequential()
    model.add(LSTM(input_shape=(timesteps, data_dim), output_dim=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=100, output_dim=1))
    # model.add(Activation("linear"))
    model.add(Activation("sigmoid"))
    # rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    sgd = optimizers.sgd(lr=0.0001, decay=1e-6,
                         momentum=0.9, nesterov=True)
    # model.compile(loss="mean_squared_logarithmic_error", optimizer=rms,
    # model.compile(loss="kullback_leibler_divergence", optimizer=rms,
    model.compile(loss="binary_crossentropy", optimizer=sgd,
                  metrics=['accuracy'])
    return model


def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=config.input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model