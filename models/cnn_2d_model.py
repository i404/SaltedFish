import keras
from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D

from util import config
from models import Model


class Cnn2DModel(Model):

    def __init__(self):
        self._epochs = 20
        self._batch_size = 16
        super().__init__()

    def _create(self):

        if self.input_shape is None:
            raise ValueError("input_shape is not set")

        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3),
                         activation='relu', padding="same",
                         input_shape=self.input_shape))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss=keras.losses.mean_absolute_error,
                      optimizer="adam",
                      # optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model
