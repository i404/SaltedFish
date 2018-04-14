from stock_reader import SequenceReader
from models import DenseModel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # print("Hello World!")
    sequence_reader = SequenceReader("data")
    targets, features = sequence_reader.load_raw_data()
    model = DenseModel()
    history = model.fit(features, targets)
    plt.plot(history.history["loss"])
    plt.show()
