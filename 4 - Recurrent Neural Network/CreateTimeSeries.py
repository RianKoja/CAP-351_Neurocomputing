# Creates one time series per ticker for training and testing an RNN

# Standard imports:
import os
import time

# PyPI imports:
import numpy as np
import pandas as pd

# Local imports:
from tools import manipulations


def create_time_series(ticker, path_source, path_dest):
    # Load prices dataset:
    prices = pd.read_csv(os.path.join(path_source, ticker + ".A.csv"))

    # Load dividends dataset:
    dividends = pd.read_csv(os.path.join(path_source, ticker + ".D.csv"))
    # Remove "action" column:
    dividends.drop(columns=["action"], inplace=True)
    dividends.columns = ["Date", "Dividend"]

    # Merge datasets based on the 'Date' column:
    merged = prices.merge(dividends, on="Date", how="left").replace(np.nan, 0)
    # Create the "goal" by shifting the price by 20 days:
    merged["Goal"] = prices["Low"].shift(-20)
    # Remove the last 20 rows:
    merged = merged.iloc[:-20]

    # Save to destination directory:
    merged.to_csv(os.path.join(path_dest, ticker + ".csv"), index=False)

    # Return dataframe for debugging purposes:
    return merged


if __name__ == "__main__":
    print("Starting", os.path.basename(__file__))
    root_path = os.path.dirname(os.path.realpath(__file__))
    tickers_list = manipulations.get_tickers()
    for Ticker in tickers_list:
        print("Ticker:", Ticker)
        try:
            create_time_series(Ticker, os.path.join(root_path, "PreProcessedData"), os.path.join(root_path, "ProcessedData"))
        except:
            # Sleep for a while then try again:
            time.sleep(30)
            try:
                create_time_series(Ticker, os.path.join(root_path, "PreProcessedData"), os.path.join(root_path, "ProcessedData"))
            except:
                pass  # maybe just not found the file

    print("Finished", os.path.basename(__file__))
