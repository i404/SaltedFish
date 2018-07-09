import tushare as ts
import os
from util import config
import pandas as pd
# -*- coding: utf-8 -*-

start_time = "2017-12-01"
index_url = f"http://quotes.money.163.com/service/chddata.html?" + \
            f"code=0000001&start={start_time.replace('-', '')}"


def get_stock_list(list_file_name):
    total = ts.get_today_all()
    with open(list_file_name, "w", encoding="utf-8") as fp:
        for index, row in total.iterrows():
            fp.write(f"{row['name']},{row['code']}\n")


def stock_name_code_from_file(list_file_name):
    with open(list_file_name, encoding="utf-8") as fp:
        for line in fp:
            name, code = line.strip().split(",")
            yield (name, code)


def get_stock_data_and_save(name, code):
    file_name = os.path.join(config.data_path, f"{code}.csv")
    df = ts.get_hist_data(code, start=start_time)
    if df is not None:
        df.to_csv(file_name)
        print("\n")
    else:
        print("\tNone")


def get_total_index():
    df = pd.read_csv(index_url, encoding="gbk")
    result_path = os.path.join("total_index.csv")
    df.to_csv(result_path, sep=",", encoding="utf-8", index=False)


if __name__ == "__main__":

    data_path = os.path.join("data")

    list_file = os.path.join("stock_list.csv")
    get_total_index()
    need_fresh = True
    if need_fresh:
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        get_stock_list(list_file)

        for name, code in stock_name_code_from_file(list_file):
            print(f"downloading {name}_{code}", end="")
            get_stock_data_and_save(name, code)
