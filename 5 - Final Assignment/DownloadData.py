# Gets a list of tickers from the "Tickers.xlsx" file and updates the data for each ticker in the Preprocessed data folder.
# Standard imports:
import datetime as dt
import os
import time

# PyPI imports:
import pandas as pd
import pandas_datareader

# Local imports:
from tools import manipulations

print("Starting", __file__)
start_time = time.time()

preprocessed_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "PreProcessedData")

# Create folder if it does not exist:
os.makedirs(preprocessed_data_folder, exist_ok=True)

tickers_list = manipulations.get_tickers()

# Now for each ticker, download data from yahoo finance: (date fixed to allow reproduction when cloning repository)
for ticker in tickers_list:
    try:
        print("Downloading", ticker)
        df_asset = pandas_datareader.data.DataReader(ticker + ".SA", "yahoo", dt.datetime(1995, 1, 1), dt.datetime(2021, 10, 16))
        df_asset.to_csv(os.path.join(preprocessed_data_folder, ticker + ".A.csv"))

    except:
        print("Failed.")

print("Finished", __file__)
print("--- %s seconds ---" % (time.time() - start_time))
