from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from keras.callbacks import EarlyStopping


class Model(object):

    def __init__(self):

        self.validation_split = 0.3
        self.batch_size = 2048

        self._metrics = ['accuracy', 'precision', 'recall']
        self._default_epochs = 100
        self._default_n_jobs = 1
        self._default_cv_num = 10

        # todo: choose better `monitor` for early stop
        self.callback = EarlyStopping(monitor='val_loss', min_delta=0,
                                      patience=60, verbose=1, mode='auto')

        self.model = self._create()

    def _create(self):
        raise NotImplementedError("create")

    def fit(self, x, y):
        history = self.model.fit(
            x, y, epochs=self.epochs, batch_size=self.batch_size,
            verbose=1, validation_split=self.validation_split, callbacks=[self.callback])
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

        skf = StratifiedShuffleSplit(n_splits=self._default_cv_num)
        results = cross_validate(
            model, x, y, cv=skf, scoring=self._metrics,
            return_train_score=True, n_jobs=self.n_jobs)

        return results

    @property
    def epochs(self):
        if hasattr(self, "_epochs"):
            return self._epochs
        else:
            return self._default_epochs

    @property
    def n_jobs(self):
        if hasattr(self, "_n_jobs"):
            return self._n_jobs
        else:
            return self._default_n_jobs

    @property
    def cv_num(self):
        if hasattr(self, "_cv_num"):
            return self._cv_num
        else:
            return self._default_cv_num

