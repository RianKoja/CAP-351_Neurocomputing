########################################################################################################################
# Creates a report explaining the current work and presenting obtained results.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports:
import glob
import os
import pickle

# PyPI imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Local imports
from tools import createdocument, PlotData, classes


# Define constants:
ticker = "EMBR3"  # "AZUL4"  # "SUZB3"  # "MGLU3" #"TAEE3"  # "AZUL4" "ENGI4"

# Start building a report:
mount_dir = os.path.join(os.path.dirname(__file__), "mount")
results_dir = os.path.join(os.path.dirname(__file__), "Results")
data_folder = os.path.join(os.path.dirname(__file__), "ProcessedData")
doc = createdocument.ReportDocument(title="Recurrent Neural Network Exercise", user_name="Rian Koja")
doc.add_heading("Introduction", level=1)
doc.add_paragraph(
    "In this work, several Long Short-Term Memory (LSTM) networks are trained on a time series extracted from yahoo finance for a ticker traded in the Brazilian stock market. The forecast attempts to use current market data of an asset to assess its lowest trading price twenty financial days later. The idea being that if the trading price of a stock could be known twenty days in advance, a buy/sell decision could be made on spot, and profits could be earned by generating accurate predictions. Using a gap of twenty days attempts to remove the fact that the markets are believed to be dominated by volatility and extreme events on a daily basis, but asset prices should be driven by their intrinsic value over longer periods of time, while in the mean term a mixture of the momentum of market dynamics and the underlying properties of the market, hence a period of twenty financial days, which is approximately one calendar month should accommodate for meaningful fluctuations, while neither attempting to predict fluctuations dominated by randomness nor attempting to guess evolutions of the long term market and bussines reality."
)

doc.add_paragraph(
    "This is a very difficult problem for real life application, since any trivial method to forecast stock prices in the short or medium term, once deployed to the real market opposes its own effectiveness, that is, purchasing an asset causes it's price to rise while selling it causes the price to drop. Since many players act on the market, if several of them employ the same technique, their execution will degrade in quality to the point of distorting the market away from behaving as the technique assumed. Despite this fact, the goal in the current project is to experiment with the smallest possible recurrent networks to address this problem."
)

doc.add_heading("Preparing Data", level=1)

doc.add_heading("Listing relevant tickers", level=2)
doc.add_paragraph(
    'A manually curated list of tickers is available in the "RawData" folder. It is read by the script "DownloadData.py", and contains tickers traded in the brazilian exchange "B3", they comprise stocks, Brazilian Depository Receipts (BDRs), Real Estate Funds (FFIs in the portuguese acronym), exchange traded funds (ETFs) and Brazilian depositary receipts of foreing exchange traded funds (BDR-ETF).'
)

doc.add_heading("Downloading Financial Data", level=2)
doc.add_paragraph(
    'The data is downloaded from Yahoo Finance, and stored in the "PreProcessedData" folder by running script "DownloadData.py". A pair of .csv files is created for each ticker, one containing exchange info, one with stock events (i.e. dividends), these file are named "[ticker name].A.csv" and "[ticker name].D.csv" respectively.'
)

doc.add_paragraph('The data in the "A" .csv file has following columns:')
doc.add_paragraph("Ticker: The ticker name of the asset.")
doc.add_paragraph("Date: The date of the trading day.")
doc.add_paragraph("Open: The opening price of the asset on that trading day.")
doc.add_paragraph("High: The highest price of the asset during that trading day.")
doc.add_paragraph("Low: The lowest price of the asset during that trading day.")
doc.add_paragraph("Close: The closing price of the asset on that trading day.")
doc.add_paragraph("Volume: The volume of the asset over that trading day.")
doc.add_paragraph("Adj Close: The adjusted closing price of the asset on that date.")


doc.add_paragraph('The data in the "D" .csv file has following columns:')
doc.add_paragraph("Date: The date of the trading day.")
doc.add_paragraph('action: The event that occurred with that asset on that date, normally just "dividend"')
doc.add_paragraph("value: The amount of the dividend paid per unit of the asset.")

doc.add_heading("Preprocessing Data", level=2)
doc.add_paragraph(
    'The data is preprocessed by the script "PreProcessData.py", which creates a .csv file for each ticker containing the data in both the previous dataframes, filling the a column with dividends with zeros for days without any events and adding a "Goal" column which represents the minimum value for which the stock traded 20 trading days after the specified Date information.'
)

doc.add_heading("Training Models", level=2)
doc.add_paragraph(
    'Using the ticker "'
    + ticker
    + '", which corresponds to a company that is neither too old nor too young, such that the size of the dataset is easy to handle while being possible to properly train, a validation set is created using the 10% most recent data, while the 10% oldest data is used for test and the rest is used for training. Several models are trained and the scored are saved in a .csv file in the "Results" folder. One of the goals in this exercise was to find the smallest possible model that would yield satisfactory results for the problem at hand. While, as discussed before, the problem is intrinsically difficult and using large models would be acceptable as long as the training and activation were feasible, due to the secondary goal of the exercise, a list of possible topologies was created with the script "manipulations.py", is located in the folder "models", named "Layers_by_complexity.csv", and it specifies the amount of parameters each topology contains if it represented a standard Multi-Layer Perceptron network (MLP), despite LSTM cells being used. This is meant to provide a simple proxy to how "big" or "small" a model is. Each model has 7 neurons in an input layer and 1 in its output layer, which are often omitted when describing the topology. Topologies from the smallest to the highest complexity were tested, but were limited to 6 inner layers with at most 25 neurons each.'
)

doc.add_paragraph('The script "PredictTicker.py" trains these models and computes the scores in all data sets, saving them in a csv file with the name of the ticker in the "Results" folder, ')

# Load a dataframe from the .csv file with the scores:
df_scores = pd.read_csv(os.path.join(results_dir, "Results_" + ticker + ".csv"))

PlotData.plot_complexity_correlation(df_scores, "MSE Train")
doc.add_fig()

PlotData.plot_complexity_correlation(df_scores, "MSE Future")
doc.add_fig()

PlotData.plot_complexity_correlation(df_scores, "MSE Past")
doc.add_fig()

# Load data used for training, testing and validation:
data_file = os.path.join(data_folder, ticker + ".csv")
mdl_empty = classes.ModelSpecs(data_file)
x, y, dt = mdl_empty.get_x_y(data_file)
x_train, y_train, dt_train, x_test_ini, y_test_ini, dt_test_ini, x_test_end, y_test_end, dt_test_end = PlotData.split_sets(x, y, dt)

doc.add_heading("Sample Results", level=2)

doc.add_heading("First Model Results", level=3)
doc.add_paragraph(f"The first model tested has the topology (7, 1, 1), that is, only a single LSTM neuron between the input and output layers. Naturally, it expected to train relatively well, but to generalize poorly. The reason for this assumption is that the optimization algorithm should be able to find a direct descent path more easily, although it might not yield the best results, nor encode any sophisticated logic.")
# Load model for ticker from saved pickle file:
with open(os.path.join(os.path.dirname(__file__), "models", ticker + "(7, 1, 1).pckl"), "rb") as fid:
    mdl = pickle.load(fid)
PlotData.plot_predictions(ticker, mdl)
doc.add_fig()
PlotData.plot_training(mdl)
doc.add_fig()

PlotData.strategy_gain_plot(x_test_ini, mdl.predict_output_ini, y_test_ini, dt_test_ini, doc, "Past/Test Data")
PlotData.strategy_gain_plot(x_train,    mdl.predict_output_trn, y_train,    dt_train,    doc,  "Training Data")
PlotData.strategy_gain_plot(x_test_end, mdl.predict_output_end, y_test_end, dt_test_end, doc, "Future/Validation Data")

with open(os.path.join(os.path.dirname(__file__), "models", ticker + "(7, 2, 1).pckl"), "rb") as fid:
    mdl = pickle.load(fid)
PlotData.plot_predictions(ticker, mdl)
doc.add_fig()
PlotData.plot_training(mdl)
doc.add_fig()

# find most recent file in models folder:
latest_model = max(glob.iglob(os.path.join(os.path.dirname(__file__), "models", ticker + "*.pckl")), key=os.path.getctime)

with open(latest_model, "rb") as fid:
    mdl = pickle.load(fid)
PlotData.plot_predictions(ticker, mdl)
doc.add_fig()
PlotData.plot_training(mdl)
doc.add_fig()

doc.add_paragraph("For training data:")
PlotData.compare_prediction(x_train, mdl.predict_output_trn, dt_train, doc)
doc.add_paragraph("For validation data (past):")
PlotData.compare_prediction(x_test_ini, mdl.predict_output_ini, dt_test_ini, doc)

PlotData.strategy_gain_plot(x_test_ini, mdl.predict_output_ini, y_test_ini, dt_test_ini, doc, "Past/Test Data")
PlotData.strategy_gain_plot(x_train,    mdl.predict_output_trn, y_train,    dt_train,    doc, "Training Data")
PlotData.strategy_gain_plot(x_test_end, mdl.predict_output_end, y_test_end, dt_test_end, doc, "Future/Validation Data")

#mdl.plot_model()

doc.finish()
# Launch the created document:
if os.name == "nt":
    os.system(f'start "" "{os.path.join(mount_dir, doc.file_name)}"')
else:
    os.system(f'xdg-open "{os.path.join(mount_dir, doc.file_name)}"')
print("Finished", os.path.basename(__file__))
