from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=100, output_dim=1))
    # model.add(Activation("linear"))
    model.add(Activation("relu"))
    rms = optimizers.RMSprop(lr=0.0002, rho=0.9, epsilon=1e-06)
    # model.compile(loss="mean_squared_logarithmic_error", optimizer=rms,
    # model.compile(loss="kullback_leibler_divergence", optimizer=rms,
    model.compile(loss="mean_squared_logarithmic_error", optimizer=rms,
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    return model


def load_data(file_path, fields, seq_len):
    df = pd.read_csv(file_path)[fields]
    sample = []
    seq_len += 1
    for index in range(len(df) - seq_len):
        # sample.append(df[index: index + seq_len])
        sample.append([[_] for _ in df[index: index + seq_len]])

    train, test = train_test_split(sample, test_size=0.3)
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train[:, :-1], train[:, -1]), (test[:, :-1], test[:, -1])


def run(batch_size=50):
    (x_train, y_train), (x_test, y_test) = load_data("sample.csv", "open", 20)

    model = create_model()

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=200,
              validation_data=(x_test, y_test))
    m = model.evaluate(x_test, y_test, batch_size=batch_size)
    return m


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
    N = 5
    metrics = [run() for i in range(N)]
    for score, mse, mae in metrics:
        print("==============")
        print(f"test score: {score}\ntest mse: {mse}\ntest mae: {mae}\n")
    metrics_array = np.array(metrics)
    print(f"metrics mean: {np.mean(metrics_array, 0)}")
    print(f"metrics var: {np.var(metrics_array, 0)}")
