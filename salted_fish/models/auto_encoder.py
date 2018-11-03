from keras import Sequential
from keras.layers import Dense, regularizers, Dropout
from sklearn.model_selection import train_test_split


class AutoEncoder(object):

    def __init__(self, encoding_dim=25):
        self.encoding_dim = encoding_dim

    def encode(self, features):
        l1_penalty = 0
        regular = regularizers.l1(l1_penalty)

        input_dim = features.shape[1]

        train, test = train_test_split(features)

        encoder = Sequential()
        encoder.add(Dense(200, input_shape=(input_dim,), activation="relu",
                          activity_regularizer=regular))
        encoder.add(Dropout(0.2))
        encoder.add(Dense(150, activation="relu",
                          activity_regularizer=regular))
        encoder.add(Dropout(0.2))
        encoder.add(Dense(self.encoding_dim, activation="relu",
                          activity_regularizer=regular))

        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(Dense(150, activation="relu",
                              activity_regularizer=regular))
        autoencoder.add(Dropout(0.2))
        autoencoder.add(Dense(200, activation="relu",
                              activity_regularizer=regular))
        autoencoder.add(Dropout(0.2))
        autoencoder.add(Dense(input_dim, activation="sigmoid",
                              activity_regularizer=regular))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(train, train,
                        epochs=25,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(test, test))

        encode_res = encoder.predict(features)
        return encode_res

