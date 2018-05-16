from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from keras.callbacks import EarlyStopping


class Model(object):

    def __init__(self, input_shape=None, validation_split=0.3,
                 batch_size=2048, epochs=100, normalize=False,
                 early_stop_epochs=None, verbose=1):

        self.validation_split = validation_split
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.epochs = epochs
        self.normalize = normalize
        self.n_jobs = 1
        self.cv_num = 10
        self.early_stop_epochs = early_stop_epochs
        self.verbose = verbose
        # self._default_batch_size = 2048
        # self.input_shape = None
        # self._default_epochs = 100
        # self._default_n_jobs = 1
        # self._default_cv_num = 10

        self._metrics = ['accuracy', 'precision', 'recall']

        # todo: choose better `monitor` for early stop
        if self.early_stop_epochs is not None:
            self.callbacks = [
                EarlyStopping(monitor='val_acc', min_delta=0,
                              patience=self.early_stop_epochs,
                              verbose=1, mode='auto')]
        else:
            self.callbacks = None

        self.model = None

    def _create(self):
        raise NotImplementedError("create")

    def fit(self, x, y):

        if self.model is None:
            self.model = self._create()

        history = self.model.fit(
            x, y, epochs=self.epochs, batch_size=self.batch_size,
            verbose=self.verbose, validation_split=self.validation_split, callbacks=self.callbacks)
        return history

    def predict(self, x):
        prob = self.predict_prob(x)
        return [1 if x > 0.5 else 0 for x in prob]

    def predict_prob(self, x):
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

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
