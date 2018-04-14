from stock_reader import SequenceReader
from models import DenseModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score

if __name__ == "__main__":
    # print("Hello World!")
    sequence_reader = SequenceReader("data")
    t_targets, t_features, v_targets, v_features = sequence_reader.load_raw_data()
    model = DenseModel()
    history = model.fit(t_features, t_targets)

    p_target = model.predict(v_features)
    for m_fun in [accuracy_score, recall_score, precision_score]:
        print(m_fun(v_targets, p_target))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()
