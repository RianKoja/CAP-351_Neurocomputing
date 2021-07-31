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
if not os.path.exists(preprocessed_data_folder):
    os.makedirs(preprocessed_data_folder)


tickers_list = manipulations.get_tickers()


# Now for each ticker, download data from yahoo finance: (date fixed to allogit puw reproduction)
for ticker in tickers_list:
    try:
        print("Downloading", ticker)
        df_asset = pandas_datareader.data.DataReader(ticker + ".SA", "yahoo", dt.datetime(1995, 1, 1), dt.datetime(2021, 9, 10))
        df_asset.to_csv(os.path.join(preprocessed_data_folder, ticker + ".A.csv"))

        # Now do the same for dividend info:
        df_dividend = pandas_datareader.data.DataReader(ticker + ".SA", "yahoo-dividends", dt.datetime(2000, 1, 1), dt.datetime(2021, 9, 10))
        # Invert order:
        df_dividend = df_dividend.iloc[::-1]
        # Label index column as "Date":
        df_dividend.index.name = "Date"
        df_dividend.to_csv(os.path.join(preprocessed_data_folder, ticker + ".D.csv"))
    except:
        print("Failed.")

print("Finished", __file__)
print("--- %s seconds ---" % (time.time() - start_time))
