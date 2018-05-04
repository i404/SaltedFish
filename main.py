from stock_reader import SequenceReader, CnnFormatReader, MatrixReader
from models import DenseModel, Cnn1DModel, Cnn2DModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score


def top_n_precision(n, probs, labels):

    res_lst = list(zip(probs, labels))
    res_lst = sorted(res_lst, key=lambda x: x[0], reverse=True)
    res_lst = res_lst[:n]
    # print(res_lst)
    if res_lst[-1][0] < 0.5:
        raise Exception(f"{n}th prob < 0.5")
    correct_cnt = sum([x[1] for x in res_lst])
    return 1.0 * correct_cnt / len(res_lst)


def evaluate_model(reader, model):

    data_dict = reader.load_raw_data()
    t_targets = data_dict.get("train_targets")
    t_features = data_dict.get("train_features")
    v_targets = data_dict.get("validation_targets")
    v_features = data_dict.get("validation_features")
    stock_ids = data_dict.get("stock_ids")
    shape = data_dict.get("shape")

    model.set_input_shape(shape)
    history = model.fit(t_features, t_targets)

    p_target = model.predict(v_features)
    for name, m_fun in [("acc", accuracy_score),
                        ("recall", recall_score),
                        ("precision", precision_score)]:
        print(f"{name}: {m_fun(v_targets, p_target)}")

    plt.plot(history.history["loss"][1:])
    plt.plot(history.history["val_loss"][1:])
    plt.show()

    p_prob = model.predict_prob(v_features)
    for n in [1, 3, 5, 10, 15, 20]:
        res = top_n_precision(n, probs=p_prob.flatten().tolist(), labels=v_targets.flatten().tolist())
        print(f"top {n} precision is {res}")


if __name__ == "__main__":
    # sequence_reader = CnnFormatReader(SequenceReader("data"))
    # model = Cnn1DModel()
    model = Cnn2DModel()
    sequence_reader = CnnFormatReader(MatrixReader("data"))
    evaluate_model(sequence_reader, model)

