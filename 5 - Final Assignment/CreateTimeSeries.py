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

    # Create the "goal" by shifting the price by 20 days:
    prices["Goal"] = prices["Low"].shift(-20)
    # Create a column "Previous Close" by shifting close price up by 1 day:
    prices["Previous Close"] = prices["Close"].shift(+1)
    # Remove first row: (would be filled with NaN)
    prices = prices.drop(labels=0, axis='index')

    # Make columns numeric:
    df = pd.DataFrame()
    df["Date"] = prices["Date"].copy()
    prices = prices.drop(labels="Date", axis='columns').apply(pd.to_numeric)

    # Multiply "Volume" by the average of "Open" and "Close" on that day (as a proxy for actual financial volume), then
    # divide by some arbitrary high number, so that normalization is unnecessary before running the ML models.
    df["Volume_Adjusted"] = 5e-11 * prices["Volume"] * (prices["Close"] + prices["Open"])
    prices.drop(labels="Volume", axis='columns', inplace=True)
    # Convert prices to daily increases:
    np.seterr(divide='ignore')
    for col in prices.columns:
        if col not in ['Previous Close']:
            df[col] = np.log(prices[col]/prices["Previous Close"])
    np.seterr(divide='warn')
    
    # Save to destination directory:
    df.to_csv(os.path.join(path_dest, ticker + ".csv"), index=False)

    # Return dataframe for debugging purposes:
    return prices


if __name__ == "__main__":
    print("Starting", os.path.basename(__file__))
    root_path = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(root_path, "PreProcessedData")
    output_path = os.path.join(root_path, "ProcessedData")
    os.makedirs(output_path, exist_ok=True)

    # Iterate over tickers:
    tickers_list = manipulations.get_tickers()
    for Ticker in tickers_list:
        print("Ticker:", Ticker)
        try:
            create_time_series(Ticker, input_path, output_path)
        except FileNotFoundError:
            # Sleep for a while then try again:
            time.sleep(30)
            try:
                create_time_series(Ticker, input_path, output_path)
            except FileNotFoundError:
                print("Failed to created timeseries for", Ticker)
                pass  # maybe just failed to download data

    print("Finished", os.path.basename(__file__))
