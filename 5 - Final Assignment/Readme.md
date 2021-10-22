## Final Assignment
This project is an improved version of the 4th assignment developed during the CAP-351 course. Goal is to approach the stock medium term forecasting problem using considerably larger data sets for training and more input features. Model size and topology is mostly limited by available training time and processing capabilities of the development machine (32gb RAM, RTX 2070, i7-9700K).

## About the Files in this repository and how to update them:

# Repository link:
https://github.com/RianKoja/CAP-351_Neurocomputing/tree/main/4%20-%20Recurrent%20Neural%20Network

# Ticker.xlsx
Go to "Fundamentus.com.br", advanced search with all empty fields and copy to excel file. Or search other sources for updated lists of tickers.

# PreProcessedData (folder)
Run the script `DownloadData.py` to populate this folder. It creates two .csv files for each ticker, one with daily trading prices and volume, another with dividends.

# ProcessedData (folder)

After updating the csv files in PreProcessedData, run `CreateTimeSeries.py` to process the data. This creates a better data archive, that can be augmented by downloading extra data and processing it. 

# models (folder)
Run `manipulations.py` to create a list of topologies ordered by complexity. Then run `PredictTicker.py`, where mid script a ticker is specified but can be modified, and the script will run the topologies to create the models.

# Results (folder)
Run `PredictTicker.py` to create the aforementioned pickled models, and .csv files with the mean squared errors obtained in training and testing. A slice of the first 10% financial days in the data is used to test the models, while a slice of the last 10% is used to validate the models. After that, running `CreateReport.py` will create a report explaining the exercise and presenting results.
