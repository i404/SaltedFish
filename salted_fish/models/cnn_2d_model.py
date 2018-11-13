import keras
from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D

from models import BasicModel
from reprocess import reshape_2d_feature_for_2d_cnn
from stock_reader import MatrixReader


class Cnn2DModel(BasicModel):

    def _create_reader(self):
        return MatrixReader(
            self.data_path, self.index_file, self.sequence_length)

    def __init__(self, epochs=20, batch_size=16,
                 early_stop_epochs=None, verbose=1):
        # self._epochs = 20
        # self._batch_size = 16
        super().__init__(epochs=epochs, batch_size=batch_size,
                         early_stop_epochs=early_stop_epochs, verbose=verbose)

    def _reshape_input(self, raw_features):
        shape, feature = reshape_2d_feature_for_2d_cnn(raw_features)
        self.input_shape = shape
        return feature

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
