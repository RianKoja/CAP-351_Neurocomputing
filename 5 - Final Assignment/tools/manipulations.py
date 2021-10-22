# Standard imports:
import os

# PyPI imports:
import numpy as np
import pandas as pd


def get_tickers():
    raw_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "RawData")
    tickers_file = os.path.join(raw_data_folder, "Tickers.xlsx")
    # Create a list of tickers from the Tickers.xlsx file:
    tickers_list = []
    file_path = os.path.join(raw_data_folder, tickers_file)
    for sheet in sorted(set(pd.ExcelFile(tickers_file).sheet_names) - {"BDR", "BDR_ETF"}):
        # Read data from excel into a dataframe without header:
        temp_data = pd.read_excel(file_path, sheet_name=sheet, header=None)
        # Add the tickers to the list:
        temp_list = temp_data[0].to_list()
        tickers_list += temp_list

    return tickers_list


def str2list(eg):
    return [int(k) for k in eg.replace("(", "").replace(")", "").replace(" ", "").split(",") if k != ""]


def split_sets(x, y, dt):
    # Separate train and test data:
    split_ini = int(x.shape[0] * 0.15)
    split_end = int(x.shape[0] * 0.85)
    x_trn = x[split_ini:split_end]
    y_trn = y[split_ini:split_end]
    dt_trn = dt.iloc[split_ini:split_end]
    x_ini = x[:split_ini]
    y_ini = y[:split_ini]
    dt_ini = dt.iloc[:split_ini]
    x_end = x[split_end:]
    y_end = y[split_end:]
    dt_end = dt.iloc[split_end:]

    return x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end


if __name__ == "__main__":
    print("Finished", os.path.basename(__file__))
