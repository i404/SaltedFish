import keras
from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense

from util import config
from models import Model


class CnnModel(Model):

    def __init__(self):
        self._epochs = 1000
        super().__init__()

    def _create(self):
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(3, 3),
                         activation='relu', padding="same",
                         input_shape=config.input_shape))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
        model.add(Dropout(0.2))
        model.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
        model.add(Dropout(0.2))
        model.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='softmax'))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model
