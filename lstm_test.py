from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=15,
#           validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)


def create_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=100, output_dim=1))
    model.add(Activation("linear"))
    rms = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
    model.compile(loss="mse", optimizer=rms, metrics=['accuracy'])
    return model


def load_data(file_path, fields, seq_len):
    df = pd.read_csv(file_path)[fields]
    sample = []
    seq_len += 1
    for index in range(len(df) - seq_len):
        # sample.append(df[index: index + seq_len])
        sample.append( [ [_] for _ in df[index: index + seq_len] ])

    train, test = train_test_split(sample, test_size=0.3)
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train[:, :-1], train[:, -1]), (test[:, :-1], test[:, -1])


(x_train, y_train), (x_test, y_test) = load_data("sample.csv", "open", 20)

model = create_model()

print('Train...')
model.fit(x_train, y_train,
          batch_size=100,
          epochs=150,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=100)
print('Test score:', score)
print('Test accuracy:', acc)
