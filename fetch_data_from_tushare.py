import tushare as ts
import os
import config
# -*- coding: utf-8 -*-


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
    df = ts.get_hist_data(code, start=config.start_time)
    if df is not None:
        df.to_csv(file_name)
        print("\n")
    else:
        print("\tNone")


if __name__ == "__main__":
    # list_file = os.path.join(config.data_path, config.stock_list_file_name)
    list_file = os.path.join(config.stock_list_file_name)
    need_fresh = True
    if need_fresh:
        if not os.path.exists(config.data_path):
            os.mkdir(config.data_path)
        get_stock_list(list_file)

        for name, code in stock_name_code_from_file(list_file):
            print(f"downloading {name}_{code}", end="")
            get_stock_data_and_save(name, code)
