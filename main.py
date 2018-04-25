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


def top_n_precision(n, probs, labels):

    res_lst = list(zip(probs, labels))
    res_lst = sorted(res_lst, key=lambda x: x[0], reverse=True)
    res_lst = res_lst[:n]
    # print(res_lst)
    if res_lst[-1][0] < 0.5:
        raise Exception(f"{n}th prob < 0.5")
    correct_cnt = sum([x[1] for x in res_lst])
    return 1.0 * correct_cnt / len(res_lst)


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

    p_prob = model.predict_prob(v_features)
    for n in [1, 3, 5, 10, 15, 20]:
        res = top_n_precision(n, probs=p_prob.flatten().tolist(), labels=v_targets.flatten().tolist())
        print(f"top {n} precision is {res}")


