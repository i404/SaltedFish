from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate


class BasicModel(object):

    def __init__(self, input_shape=None,
                 learning_rate=0.001, batch_size=32, epochs=100,
                 early_stop_epochs=None,
                 data_path=None, index_file=None,
                 sequence_length=32, verbose=1):

        self.loss = "binary_crossentropy"
        self.loss_weights = None

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stop_epochs = early_stop_epochs
        self.verbose = verbose

        self.data_path = data_path
        self.index_file = index_file
        self.sequence_length = sequence_length
        self.reader = None

        self.callbacks = []
        if self.early_stop_epochs is not None:
            early_stop = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=self.early_stop_epochs,
                verbose=1,
                mode='auto',
                restore_best_weights=True)
            self.callbacks.append(early_stop)

        self.model = None

    def model_name(self):
        return str(self).split(" ")[0].split(".")[-1]

    def get_reader(self):
        if self.reader is None:
            self.reader = self._create_reader()
        return self.reader

    def _create_reader(self):
        raise NotImplementedError("create_reader")

    def _create(self):
        raise NotImplementedError("create")

    def _reshape_input(self, features):
        raise NotImplementedError("reshape_input")

    def _reshape_target(self, targets):
        return targets

    def fit(self, x, y, test_x, test_y):

        x = self._reshape_input(x)
        test_x = self._reshape_input(test_x)
        y = self._reshape_target(y)
        test_y = self._reshape_target(test_y)

        if self.model is None:
            self.model = self._create()
            self.model.summary()

        adam = Adam(lr=self.learning_rate)
        self.model.compile(
            optimizer=adam,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=['accuracy'])

        history = self.model.fit(
            x, y, epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=(test_x, test_y),
            callbacks=self.callbacks)
        return history

    def predict(self, x):
        prob = self.predict_prob(x)
        return [1 if x > 0.5 else 0 for x in prob]

    def predict_prob(self, x):
        x = self._reshape_input(x)
        return self.model.predict(x)
