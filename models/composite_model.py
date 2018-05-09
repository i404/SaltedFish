from models import Model


class CompositeModel(Model):

    def __init__(self, model_lst=None):
        if model_lst is None:
            self.models = []
        else:
            self.models = model_lst

    def add_model(self, model):
        self.models.append(model)

    def fit(self, x, y):
        history = []
        for model in self.models:
            tmp_history = model.fit(x, y)
            if isinstance(tmp_history, list):
                history += tmp_history
            else:
                history.append(tmp_history)
        return history

    def predict_prob(self, x):
        probs = []
        for model in self.models:
            tmp_prob = model.predict_prob(x)
            if isinstance(tmp_prob, list):
                probs += tmp_prob
            else:
                probs.append(tmp_prob)
        return sum(probs) / len(probs)

    def set_input_shape(self, input_shape):
        for model in self.models:
            model.set_input_shape(input_shape)
