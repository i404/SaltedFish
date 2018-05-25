from stock_reader import SequenceReader, CnnFormatReader, MatrixReader
from models import DenseModel, Cnn1DModel, Cnn2DModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import time


def top_n_precision(y_pred, y_true, ids):

    pred_label_id_lst = list(zip(y_pred, y_true, ids))
    sorted_lst = sorted(pred_label_id_lst, key=lambda x: x[0], reverse=True)

    for pred, label, s_id in sorted_lst[:10]:
        print(f"pred_prob: {pred}, label: {label}, stock_id: {s_id}")

    for n in [1, 3, 5, 10, 15, 20]:
        res_lst = sorted_lst[:n]
        if res_lst[-1][0] < 0.5:
            raise Exception(f"{n}th prob < 0.5")
        correct_cnt = sum([x[1] for x in res_lst])
        res = 1.0 * correct_cnt / len(res_lst)
        print(f"top {n} precision is {res}")


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
    p_pred_lst = p_prob.flatten().tolist()
    v_targets_lst = v_targets.flatten().tolist()
    top_n_precision(p_pred_lst, v_targets_lst, stock_ids)


def evaluate_composite_models(reader_models):

    total_probs = []
    v_targets = None
    stock_ids = None

    for reader, model in reader_models:

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
        # plt.show()

        tmp_p_prob = model.predict_prob(v_features)
        # with open(f"model_res_{int(time.time())}", "w") as fp:
        #     for s_id, y, p in zip(stock_ids,
        #                           v_targets.tolist(),
        #                           tmp_p_prob.flatten().tolist()):
        #         fp.write(f"{s_id},{p},{y}\n")

        total_probs.append(tmp_p_prob.flatten())

    # p_preds = sum(np.array(total_probs)) / len(total_probs)
    p_preds = combine_models_result(total_probs)
    p_target = [1 if x > 0.5 else 0 for x in p_preds]
    print()
    for name, m_fun in [("acc", accuracy_score),
                        ("recall", recall_score),
                        ("precision", precision_score)]:
        print(f"{name}: {m_fun(v_targets, p_target)}")
    print()
    v_targets_lst = v_targets.flatten().tolist()
    top_n_precision(p_preds, v_targets_lst, stock_ids)

    plt.show()


def combine_models_result(total_probs):

    probs = np.array(total_probs)
    model_num, rows = probs.shape
    res_label = []
    for i in range(0, rows):
        ith_prob = sum(probs[:, i]) / model_num
        ith_label_sum = sum([1 if x > 0.5 else 0 for x in probs[:, i]])
        if ith_label_sum == model_num:
            res_label.append(ith_prob)
        else:
            res_label.append(ith_prob * 0.1)
    return res_label


if __name__ == "__main__":

    data_path = "data"
    verbose = 0

    models_lst = [
        (CnnFormatReader(SequenceReader(data_path), cnn_dim=1),
         Cnn1DModel(batch_size=2048, epochs=300,
                    early_stop_epochs=40, verbose=verbose)),

        (CnnFormatReader(MatrixReader(data_path, cols=["p_change", "turnover"]),
                         cnn_dim=1),
         Cnn1DModel(batch_size=2048, epochs=300,
                    early_stop_epochs=40, verbose=verbose)),

        (CnnFormatReader(MatrixReader(data_path), cnn_dim=2),
         Cnn2DModel(batch_size=32, epochs=15, early_stop_epochs=4, verbose=verbose))
    ]

    for feature_reader, model in [models_lst[0], models_lst[1]]:
        # feature_reader, model = models_lst[1]
        evaluate_model(feature_reader, model)

    # evaluate_composite_models([models_lst[0], models_lst[1]])

