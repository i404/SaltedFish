from stock_reader import SequenceReader, CnnFormatReader, MatrixReader
from models import DenseModel, Cnn1DSingleChannelModel, Cnn2DModel, Cnn1DMultiChannelModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import time


def timer(fun):

    def tmp(reader, model):
        beg_time = time.time()
        print(f"start at {beg_time}")
        res = fun(reader, model)
        end_time = time.time()
        print(f"end at {end_time}")
        cost_time = end_time - beg_time
        hour = int(cost_time / 60.0 / 60.0)
        minute = int((cost_time - 60 * 60 * hour) / 60.0)
        second = cost_time - 60 * 60 * hour - 60 * minute
        print(f"run time: {hour}h {minute}m {second}s (total {cost_time}s)")
        return res

    return tmp


def top_n_precision(y_pred, y_true, ids):

    pred_label_id_lst = list(zip(y_pred, y_true, ids))
    sorted_lst = sorted(pred_label_id_lst, key=lambda x: x[0], reverse=True)

    for pred, label, s_id in sorted_lst[:10]:
        print(f"pred_prob: {pred}, label: {label}, stock_id: {s_id}")

    for n in [1, 3, 5, 10, 15, 20]:
        res_lst = sorted_lst[:n]
        if res_lst[-1][0] < 0.5:
            print(f"{n}th prob < 0.5")
            # raise Exception(f"{n}th prob < 0.5")
        correct_cnt = sum([x[1] for x in res_lst])
        res = 1.0 * correct_cnt / len(res_lst)
        print(f"top {n} precision is {res}")


@timer
def evaluate_model(reader, model):

    data_dict = reader.load_raw_data()
    t_targets = data_dict.get("train_targets")
    t_features = data_dict.get("train_features")
    v_targets = data_dict.get("validation_targets")
    v_features = data_dict.get("validation_features")
    stock_ids = data_dict.get("stock_ids")
    shape = data_dict.get("shape")
    print(f"data length: {len(t_targets)}")

    model.set_input_shape(shape)
    history = model.fit(t_features, t_targets)

    p_target = model.predict(v_features)
    for name, m_fun in [("acc", accuracy_score),
                        ("recall", recall_score),
                        ("precision", precision_score)]:
        print(f"{name}: {m_fun(v_targets, p_target)}")

    plt.plot(history.history["loss"][1:])
    plt.plot(history.history["val_loss"][1:])
    # plt.plot(history.history["acc"])
    # plt.plot(history.history["val_acc"])

    print(history.history["acc"][-10:])
    print(history.history["val_acc"][-10:])
    print(history.history["loss"][-10:])
    print(history.history["val_loss"][-10:])
    # plt.show()

    p_prob = model.predict_prob(v_features)
    p_pred_lst = p_prob.flatten().tolist()
    v_targets_lst = v_targets.flatten().tolist()

    p_positive = sum([1 if x > 0.5 else 0 for x in p_pred_lst])
    v_positive = sum([1 if x > 0.5 else 0 for x in v_targets_lst])
    print(f"#positive_prob: {p_positive}, #positive_y: {v_positive}, #total_stocks: {len(v_targets_lst)}")

    top_n_precision(p_pred_lst, v_targets_lst, stock_ids)


if __name__ == "__main__":

    # data_path = "data_20180531"
    data_path = "data"
    verbose = 2

    models_lst = [
        (CnnFormatReader(SequenceReader(data_path), cnn_dim=1),
         Cnn1DSingleChannelModel(batch_size=512, epochs=400,
                                 early_stop_epochs=40, verbose=verbose)),

        (CnnFormatReader(MatrixReader(data_path, cols=["p_change", "turnover"]),
                         cnn_dim=1),
         Cnn1DMultiChannelModel(batch_size=2048, epochs=300,
                                early_stop_epochs=40, verbose=verbose)),

        (CnnFormatReader(MatrixReader(data_path), cnn_dim=2),
         Cnn2DModel(batch_size=32, epochs=15, early_stop_epochs=4, verbose=verbose))
    ]

    # for reader, model in [models_lst[0], models_lst[1]]:
    for reader, model in [models_lst[0]]:
        # start_time = time.time()
        evaluate_model(reader, model)
        # end_time = time.time()
        # print(f"train time: {end_time - start_time}")

    plt.show()
