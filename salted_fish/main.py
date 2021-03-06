"""
main of salted fish
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tensorflow import logging

from models import Cnn1DMultiChannelModel, CnnWithEmbedding, \
    CnnWithSingleDayStatus, CnnWithStatusAutoEncode, \
    CnnWithSingleDayStatusEncode
from util import timer
from util.utils import save_figure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.set_verbosity(logging.WARN)


def top_n_precision(y_pred, y_true, codes):
    pred_label_code_lst = list(zip(y_pred, y_true, codes))
    sorted_lst = sorted(pred_label_code_lst, key=lambda x: x[0], reverse=True)

    for pred, label, s_code in sorted_lst[:10]:
        print(f"pred_prob: {pred}, label: {label}, stock_id: {s_code}")

    for n in [1, 3, 5, 10, 15, 20]:
        res_lst = sorted_lst[:n]
        if res_lst[-1][0] < 0.5:
            # raise Exception(f"{n}th prob < 0.5")
            print(f"{n}th prob < 0.5")
        correct_cnt = sum([x[1] for x in res_lst])
        res = 1.0 * correct_cnt / len(res_lst)
        print(f"top {n} precision is {res}")


@timer
def evaluate_model(model):
    data_dict = model.get_reader().load_raw_data()
    train_targets = data_dict.get("train_targets")
    train_features = data_dict.get("train_features")
    test_targets = data_dict.get("test_targets")
    test_features = data_dict.get("test_features")
    v_targets = data_dict.get("validation_targets")
    v_features = data_dict.get("validation_features")
    stock_codes = data_dict.get("stock_codes")
    print(f"data length: {len(train_targets)}")

    history = model.fit(train_features, train_targets,
                        test_features, test_targets)

    def print_performance(p_type, features, targets):
        p_targets = model.predict(features)
        print(f"{p_type}_positive_rate: {1.0 * sum(targets) / len(targets)}",
              end="\t")
        for metric_name, m_fun in [("acc", accuracy_score),
                                   ("recall", recall_score),
                                   ("precision", precision_score)]:
            print(f"{p_type}_{metric_name}: {m_fun(targets, p_targets)}",
                  end="\t")
        print()

    print_performance("train", train_features, train_targets)
    print_performance("test", test_features, test_targets)
    print_performance("validation", v_features, v_targets)

    plt.plot(history.history["loss"][1:])
    plt.plot(history.history["val_loss"][1:])

    trail_name = model.model_name()
    save_figure(plt, trail_name, "fig")

    p_prob = model.predict_prob(v_features)
    p_pred_lst = p_prob.flatten().tolist()
    v_targets_lst = v_targets.flatten().tolist()

    p_positive = sum([1 if x > 0.5 else 0 for x in p_pred_lst])
    v_positive = sum([1 if x > 0.5 else 0 for x in v_targets_lst])
    print(f"#positive_prob: {p_positive}\n"
          f"#positive_y: {v_positive}\n"
          f"#total_stocks: {len(v_targets_lst)}\n"
          f"#positive fraction: {1.0 * v_positive / len(v_targets_lst)}")

    top_n_precision(p_pred_lst, v_targets_lst, stock_codes)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='process stock data')
    parser.add_argument("-v", "--verbose", help="verbose of keras",
                        type=int, default=2)

    parser.add_argument("-i", "--input_data", help="path of stock data",
                        type=str, default="data_test")

    args = parser.parse_args()

    index_file = "total_index.csv"
    data_path = args.input_data
    sequence_length = 32
    verbose = args.verbose

    def create_model(l1_lambda):
        # l1_lambda = 1e-3
        dropout = 0.0
        model = CnnWithSingleDayStatusEncode(
            cnn_filter_nums=[256, 128, 64, 32],
            cnn_dropout=[0.0, 0.0, 0.0, 0.0],
            cnn_regularize=[l1_lambda, l1_lambda, l1_lambda, l1_lambda],
            cnn_kernel_size=3,
            cnn_feature_num=512,
            single_day_change_status_embedding_dim=256,
            status_embedding_regularize=l1_lambda,
            dense_layer_nodes=[1024, 512],
            dense_layer_dropout=[dropout, dropout],
            dense_regularize=[l1_lambda, l1_lambda],
            learning_rate=1e-5,
            epochs=50,
            early_stop_epochs=3,
            batch_size=32,
            data_path=data_path,
            index_file=index_file,
            sequence_length=sequence_length,
            verbose=verbose)
        return model

    # for reader, model in [models_lst[0], models_lst[1]]:
    # for d in [0.05, 0.1, 0.15, 0.2]:
    for l1 in [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]:
        evaluate_model(create_model(l1))


if __name__ == "__main__":
    main()
