from stock_reader import SequenceReader, CnnFormatReader, MatrixReader
from models import Cnn1DSingleChannelModel, Cnn1DMultiChannelModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
import time
import os
from tensorflow import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.WARN)


def timer(fun):
    def tmp(name, reader, model):
        beg_time = time.time()
        print(f"start at {beg_time}")
        res = fun(name, reader, model)
        end_time = time.time()
        print(f"end at {end_time}")
        cost_time = end_time - beg_time
        hour = int(cost_time / 60.0 / 60.0)
        minute = int((cost_time - 60 * 60 * hour) / 60.0)
        second = cost_time - 60 * 60 * hour - 60 * minute
        print(f"run time: {hour}h {minute}m {second}s (total {cost_time}s)")
        return res

    return tmp


def save_figure(file_name, path="fig"):
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = f"{file_name}_{int(time.time())}.png"
    plt.savefig(os.path.join(path, file_name))
    plt.clf()


def top_n_precision(y_pred, y_true, ids):
    pred_label_id_lst = list(zip(y_pred, y_true, ids))
    sorted_lst = sorted(pred_label_id_lst, key=lambda x: x[0], reverse=True)

    for pred, label, s_id in sorted_lst[:10]:
        print(f"pred_prob: {pred}, label: {label}, stock_id: {s_id}")

    for n in [1, 3, 5, 10, 15, 20]:
        res_lst = sorted_lst[:n]
        if res_lst[-1][0] < 0.5:
            # raise Exception(f"{n}th prob < 0.5")
            print(f"{n}th prob < 0.5")
        correct_cnt = sum([x[1] for x in res_lst])
        res = 1.0 * correct_cnt / len(res_lst)
        print(f"top {n} precision is {res}")


@timer
def evaluate_model(trail_name, reader, model):
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

    def print_performance(p_type, features, targets):
        p_targets = model.predict(features)
        for name, m_fun in [("acc", accuracy_score),
                            ("recall", recall_score),
                            ("precision", precision_score)]:
            print(f"{p_type}_{name}: {m_fun(targets, p_targets)}", end="\t")
        print()

    print_performance("train", t_features, t_targets)
    print_performance("validation", v_features, v_targets)

    plt.plot(history.history["loss"][1:])
    plt.plot(history.history["val_loss"][1:])
    save_figure(trail_name, "fig")

    p_prob = model.predict_prob(v_features)
    p_pred_lst = p_prob.flatten().tolist()
    v_targets_lst = v_targets.flatten().tolist()

    p_positive = sum([1 if x > 0.5 else 0 for x in p_pred_lst])
    v_positive = sum([1 if x > 0.5 else 0 for x in v_targets_lst])
    print(f"#positive_prob: {p_positive}\n"
          f"#positive_y: {v_positive}\n"
          f"#total_stocks: {len(v_targets_lst)}\n"
          f"#positive fraction: {1.0 * v_positive / len(v_targets_lst)}")

    top_n_precision(p_pred_lst, v_targets_lst, stock_ids)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='process stock data')
    parser.add_argument("-v", "--verbose", help="verbose of keras",
                        type=int, default=0)

    parser.add_argument("-i", "--input_data", help="path of stock data",
                        type=str, default="data_test")

    args = parser.parse_args()

    index_file = "total_index.csv"
    data_path = args.input_data
    verbose = args.verbose

    models_lst = [
        ("single_channel_cnn_32",
         CnnFormatReader(SequenceReader(data_path, index_file, 32), cnn_dim=1),
         Cnn1DSingleChannelModel(batch_size=32, epochs=600, min_iter_num=10,
                                 early_stop_epochs=10, verbose=verbose)),

        ("multi_channel_cnn_32",
         CnnFormatReader(MatrixReader(data_path, index_file, 32), cnn_dim=1),
         Cnn1DMultiChannelModel(batch_size=32, epochs=600, min_iter_num=10,
                                early_stop_epochs=10, verbose=verbose)),
    ]

    # for reader, model in [models_lst[0], models_lst[1]]:
    for name, reader, model in models_lst:
        evaluate_model(name, reader, model)
