from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score
from models import create_dense_model, create_lstm_model, create_cnn_model

from stock_data import load_all_seq, load_all_for_cnn

import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(x, y, create_model, batch_size=100):
    model = KerasClassifier(build_fn=create_model, epochs=10,
                            batch_size=batch_size, verbose=1)
    # TODO: use time-series aware cross-validation scheme
    skf = StratifiedShuffleSplit(n_splits=10)
    results = cross_validate(
        model, x, y, cv=skf,
        scoring=['accuracy', 'precision', 'recall'],
        return_train_score=True, n_jobs=2)
    # model.evaluate()

    return results


def baseline(y):
    y_true = y[1:]
    y_pred = y[:-1]
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return acc, recall, precision


def evaluate_dense_model():
    data_set = load_all_seq("data", "p_change", 20)
    x = data_set[:, :-1]
    y = data_set[:, -1].flatten()
    y = [1 if i > 0 else 0 for i in y]
    # x = pd.read_csv("feature_encode.csv", header=None).values
    # y = pd.read_csv("target.csv", header=None).values
    # metrics = evaluate_model(x.reshape(x.shape[0], x.shape[1]),
    metrics = evaluate_model(x, y, create_model=create_dense_model)
    # metrics = evaluate_model(x, y, create_model=create_lstm_model)
    return metrics


def evaluate_cnn_model():
    targets, features = load_all_for_cnn("data_test")
    # features = pd.read_csv("feature_encode.csv", header=None).values
    # features = features.reshape(features.shape[0], 8, 8)
    # targets = pd.read_csv("target.csv", header=None)
    metrics = evaluate_model(features, targets, create_model=create_cnn_model)
    return metrics


def evaluate_model_main():
    metrics = evaluate_dense_model()
    # metrics = evaluate_cnn_model()
    print("===================")
    print(metrics)
    for key, val in metrics.items():
        print(f"{key}: {val.mean()}({val.std()})")
    # print(f"baseline: {baseline(y)}")


def plot_model_history(x, y, create_model):
    model = create_model()
    history = model.fit(x, y, validation_split=0.33, epochs=100, batch_size=100, verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()


if __name__ == "__main__":
    # x = pd.read_csv("feature_encode.csv", header=None).values
    # y = pd.read_csv("target.csv", header=None).values
    # plot_model_history(x, y, create_dense_model)

    data_set = load_all_seq("data", "p_change", 20)
    x = data_set[:, :-1]
    y = data_set[:, -1].flatten()
    y = [1 if i > 0 else 0 for i in y]
    plot_model_history(x, y, create_dense_model)

