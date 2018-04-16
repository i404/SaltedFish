from stock_reader import SequenceReader
from models import DenseModel, Cnn1DModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K


def reshape_1d_feature_depend_on_backend(raw_features):
    rows, cols = raw_features.shape
    if K.image_data_format() == 'channels_first':
        feature = raw_features.reshape(rows, 1, cols)
        shape = (1, cols)
    else:
        feature = raw_features.reshape(rows, cols, 1)
        shape = (cols, 1)
    return shape, feature


if __name__ == "__main__":
    # print("Hello World!")
    sequence_reader = SequenceReader("data")
    t_targets, t_features, v_targets, v_features = sequence_reader.load_raw_data()

    # scale = MinMaxScaler()
    # t_features = scale.fit_transform(t_features)
    # v_features = scale.transform(v_features)

    # model = DenseModel()

    tmp_shape, t_features = reshape_1d_feature_depend_on_backend(t_features)
    _, v_features = reshape_1d_feature_depend_on_backend(v_features)
    model = Cnn1DModel(tmp_shape)
    history = model.fit(t_features, t_targets)

    p_target = model.predict(v_features)
    for name, m_fun in [("acc", accuracy_score),
                        ("recall", recall_score),
                        ("precision", precision_score)]:
        print(f"{name}: {m_fun(v_targets, p_target)}")

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
