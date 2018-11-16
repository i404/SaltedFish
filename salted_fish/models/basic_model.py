from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate


class BasicModel(object):

    def __init__(self, input_shape=None, validation_split=0.3,
                 learning_rate=0.001, batch_size=32, epochs=100,
                 early_stop_epochs=None,
                 data_path=None, index_file=None,
                 sequence_length=32, verbose=1):

        self.validation_split = validation_split
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        # self.normalize = normalize
        self.n_jobs = 1
        self.cv_num = 10
        self.early_stop_epochs = early_stop_epochs
        self.verbose = verbose
        # self._default_batch_size = 2048
        # self.input_shape = None
        # self._default_epochs = 100
        # self._default_n_jobs = 1
        # self._default_cv_num = 10

        self.data_path = data_path
        self.index_file = index_file
        self.sequence_length = sequence_length
        self.reader = None

        self._metrics = ['accuracy', 'precision', 'recall']

        if self.early_stop_epochs is not None:
            self.callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=self.early_stop_epochs,
                    verbose=1,
                    mode='auto',
                    restore_best_weights=True)
            ]
        else:
            self.callbacks = None

        self.model = None

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

    def fit(self, x, y, test_x, test_y):

        x = self._reshape_input(x)
        test_x = self._reshape_input(test_x)

        if self.model is None:
            self.model = self._create()

        history = self.model.fit(
            x, y, epochs=self.epochs, batch_size=self.batch_size,
            verbose=self.verbose,
            # validation_split=self.validation_split,
            validation_data=(test_x, test_y),
            callbacks=self.callbacks)
        return history

    def predict(self, x):
        prob = self.predict_prob(x)
        return [1 if x > 0.5 else 0 for x in prob]

    def predict_prob(self, x):
        x = self._reshape_input(x)
        return self.model.predict(x)

    def evaluate(self, x, y):
        model = KerasClassifier(
            build_fn=self._create, epochs=self.epochs,
            batch_size=self.batch_size, verbose=1)

        skf = StratifiedShuffleSplit(n_splits=self.cv_num)
        results = cross_validate(
            model, x, y, cv=skf, scoring=self._metrics,
            return_train_score=True, n_jobs=self.n_jobs)

        return results
