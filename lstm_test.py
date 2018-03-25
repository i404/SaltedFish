from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import numpy as np
from customized_loss import binary_crossentropy, bias_loss

import os


def create_baseline_model(time_steps=20):
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


def load_all_data(path, fields, seq_len):
    res = None
    for file_name in os.listdir(path):
        tmp_arr = load_data(os.path.join(path, file_name), fields, seq_len)
        if res is None:
            res = tmp_arr
        elif not tmp_arr.size == 0:
            res = np.vstack((res, tmp_arr))
    return res


def load_data(file_path, fields, seq_len):
    df = pd.read_csv(file_path)[fields]
    data_set = []
    seq_len += 1
    for index in range(len(df) - seq_len):
        # sample.append(df[index: index + seq_len])
        # data_set.append([[_] for _ in df[index: index + seq_len]])
        data_set.append(df[index: index+seq_len].values)
    return np.array(data_set)


def evaluate_model(x, y, create_model, batch_size=100):
    model = KerasClassifier(build_fn=create_model, epochs=25,
                            batch_size=batch_size, verbose=1)
    """
    While i.i.d. data is a common assumption in machine learning theory, 
    it rarely holds in practice. If one knows that the samples have 
    been generated using a time-dependent process, it’s safer 
    to use a time-series aware cross-validation scheme 
    
    Similarly if we know that the generative process has a group structure
    (samples from collected from different subjects, experiments, measurement devices) 
    it safer to use group-wise cross-validation.
    """
    # TODO: use time-series aware cross-validation scheme
    results = cross_validate(
        model, x, y, cv=5,
        scoring=['accuracy', 'precision', 'recall'],
        return_train_score=True, n_jobs=3)
    # model.evaluate()

    return results


def baseline(y):
    y_true = y[1:]
    y_pred = y[:-1]
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return acc, recall, precision


if __name__ == "__main__":

    data_set = load_all_data("data", "p_change", 20)
    x = data_set[:, :-1]
    y = data_set[:, -1].flatten()
    y = [1 if i > 0 else 0 for i in y]
    # metrics = evaluate_model(x.reshape(x.shape[0], x.shape[1]),
    metrics = evaluate_model(x, y, create_model=create_baseline_model)
    # metrics = evaluate_model(x, y, create_model=create_lstm_model)
    print("===================")
    print(metrics)
    for key, val in metrics.items():
        print(f"{key}: {val.mean()}({val.std()})")
    print(f"baseline: {baseline(y)}")
