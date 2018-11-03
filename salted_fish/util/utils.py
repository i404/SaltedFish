import time
import os


def save_figure(plt, file_name, path="fig"):
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = f"{file_name}_{int(time.time())}.png"
    plt.savefig(os.path.join(path, file_name))
    plt.clf()


def timer(fun):
    def tmp(*args, **kwargs):
        beg_time = time.time()
        print(f"start at {beg_time}")
        res = fun(*args, **kwargs)
        end_time = time.time()
        print(f"end at {end_time}")
        cost_time = end_time - beg_time
        hour = int(cost_time / 60.0 / 60.0)
        minute = int((cost_time - 60 * 60 * hour) / 60.0)
        second = cost_time - 60 * 60 * hour - 60 * minute
        print(f"run time: {hour}h {minute}m {second}s (total {cost_time}s)")
        return res

    return tmp

