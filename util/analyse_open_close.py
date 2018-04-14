import pandas as pd

from stock_data import load_all
import matplotlib.pyplot as plt


def load_open_close_diff(file_name):
    df = pd.read_csv(file_name)[["open", "close"]]
    if len(df) <= 10:
        return None
    else:
        res = []
        op, cl = None, None
        for index, row in df.iterrows():
            if index == 0:
                op = row["open"]
            else:
                cl = row["close"]
                diff = (op - cl) / cl
                op = row["open"]
                if abs(diff) > 0.15:
                    print(f"{file_name}: {index}")

                res.append(diff)
        return res


def combine(a, b):
    if a and b:
        return a + b
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return None


def read_open_close_to_file(res_file):
    data_path = "./data"
    res = load_all(data_path, load_open_close_diff, combine)
    with open(res_file, "w") as fp:
        for t in res:
            fp.write(f"{t}\n")


if __name__ == "__main__":
    res_file_name = "tmp.csv"

    # read_open_close_to_file(res_file_name)

    with open(res_file_name) as fp:
        xs = [float(s.strip()) for s in fp]

    upper_bound = 0.10
    lower_bound = -0.10
    p = [x for x in xs if lower_bound <= x <= upper_bound]
    print(f"total points number is {len(xs)}, "
          f"[{lower_bound}, {upper_bound}] number {len(p)}")

    len_positive = len([i for i in xs if i >= 0])
    print(f"{1.0 * len_positive / len(xs)}")

    plt.hist(p, bins=40)
    plt.axvline(x=0, color="red")
    plt.axvline(x=-0.1, color="green")
    plt.axvline(x=0.1, color="green")
    plt.show()
