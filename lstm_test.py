from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate


def create_baseline_model(time_steps=20):
    """
    model with 3 dense layers
    :return: model
    """
    model = Sequential()
    first_layer = Dense(20, input_shape=(time_steps,),
                        activation="relu")
    model.add(first_layer)
    model.add(Dropout(0.2))
    second_layer = Dense(50, input_shape=(time_steps,),
                         activation="relu")
    model.add(second_layer)
    model.add(Dropout(0.2))
    third_layer = Dense(1, input_shape=(time_steps,),
                        activation="relu")
    model.add(third_layer)
    sgd = optimizers.sgd(lr=0.0001, decay=1e-6,
                         momentum=0.9, nesterov=True)
    model.compile(loss="mean_squared_error", optimizer=sgd,
                  metrics=["mean_squared_error", "mean_absolute_error"])
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
    model.add(Activation("linear"))
    rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    # model.compile(loss="mean_squared_logarithmic_error", optimizer=rms,
    # model.compile(loss="kullback_leibler_divergence", optimizer=rms,
    model.compile(loss="mean_squared_error", optimizer=rms,
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    return model


def load_data(file_path, fields, seq_len):
    df = pd.read_csv(file_path)[fields]
    data_set = []
    seq_len += 1
    for index in range(len(df) - seq_len):
        # sample.append(df[index: index + seq_len])
        data_set.append([[_] for _ in df[index: index + seq_len]])
    return np.array(data_set)


def evaluate_model(x, y, create_model, batch_size=50):
    model = KerasRegressor(build_fn=create_model, epochs=200,
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
        model, x, y, cv=10,
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error'],
        return_train_score=True, n_jobs=3)
    # model.evaluate()

    return results


if __name__ == "__main__":
    """
    linear activation: mse ~ N(10.09, 2.10)
    relu activation: mse ~ N(2.30, 0.134)
                     mse ~ N(1.83, 0.66) mae ~ N(0.61, 0.006)
    
    add another relu after 1st lstm: 
        mse ~ N(2.99, 0.06)  mae ~ N(0.787, 0.002)
    
    use sigmoid as output activation
    and use  mean_squared_logarithmic_error as loss to deal with exp in sigmoid
        mse ~ N(3.10, 1.30) mae ~ N(0.830, 0.018)
    maybe not converge modify lr to 0.0005:
        mse ~ N(1.50, 1.01) mae ~ N(0.57, 0.0030)
    lr = 0.0002, epochs = 200
        mse ~ N(1.11, 0.11)  mae ~ N(0.44, 1.2e-04)
    remove 1st relu:
        mse ~ N(1.08, 0.22) mae ~ N(0.49, 7.5e-3)
    
    """
    data_set = load_data("sample.csv", "open", 20)
    x = data_set[:, :-1]
    y = data_set[:, -1].flatten()
    metrics = evaluate_model(x.reshape(x.shape[0], x.shape[1]),
                             y, create_model=create_baseline_model)
    # metrics = evaluate_model(x, y, create_model=create_lstm_model)
    print("===================")
    print(metrics)
    for key, val in metrics.items():
        print(f"{key}: {val.mean()}({val.std()})")
