
class Model(object):

    def fit(self, x, y):
        raise NotImplementedError("fit")

    def predict(self, x):
        # raise NotImplementedError("predict")
        prob = self.predict_prob(x)
        return [1 if x > 0.5 else 0 for x in prob]

    def predict_prob(self, x):
        raise NotImplementedError("predict_prob")

    def set_input_shape(self, input_shape):
        raise NotImplementedError("set_input_shape")
