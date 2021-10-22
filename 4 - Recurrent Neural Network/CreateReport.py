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

doc.add_heading("Abstract", level=1)
doc.add_paragraph("This report presents the results of an exercise on Recurrent Neural Networks, where models of minimal size are employed on a stock forecasting problem. An introduction to this problem is given, along with some practical considerations. The dataset is defined and its preparation is explained. Several small topologies are tested, and the results are presented both in an aggregate sense and in detail for sample models. An investment strategy based on these predictions is simulated, which presents low to negative returns. Final notes are given in the conclusion.")

doc.add_heading("Introduction", level=1)
doc.add_paragraph(
    "In this work, several Long Short-Term Memory (LSTM) networks are trained on a time series extracted from Yahoo Finance for a ticker traded in the Brazilian stock market. The forecast attempts to use current market data of an asset to predict its lowest trading price twenty financial days later. The idea being that if the trading price of a stock could be known twenty days in advance, a buy/sell decision could be made on spot, and profits could be earned by generating accurate enough predictions. Using a gap of twenty days attempts to remove the fact that the markets are believed to be dominated by volatility and extreme events on a daily basis, but asset prices should be driven by their intrinsic value over longer periods of time, while in the mean term a mixture of the momentum of market dynamics and the underlying properties of the market, hence a period of twenty financial days, which is approximately one calendar month should accommodate for meaningful fluctuations, while neither attempting to predict fluctuations dominated by randomness nor attempting to guess evolutions of the long term market and business reality."
)

doc.add_paragraph(
    "Stock forecasting is a very difficult problem for real life applications, since any trivial method to forecast stock prices in the short or medium term, once deployed to the real market opposes its own effectiveness, that is, purchasing an asset causes it's price to rise while selling it causes the price to drop. Since many players act on the market, if several of them employ the same technique, their execution will degrade in quality to the point of distorting the market away from behaving as the technique assumed. Despite this fact, the goal in the current project is to experiment with the smallest possible recurrent networks to address this problem."
)

doc.add_heading("Preparing Data", level=1)

doc.add_heading("Listing relevant tickers", level=2)
doc.add_paragraph(
    'A manually curated list of tickers is available in the "RawData" folder. It is read by the script "DownloadData.py", and contains tickers traded in the Brazilian exchange "B3", they comprise stocks, Brazilian Depository Receipts (BDRs), Real Estate Funds (FFIs in the Portuguese acronym), exchange traded funds (ETFs) and Brazilian depositary receipts of foreign exchange traded funds (BDR-ETF).'
)

doc.add_heading("Downloading Financial Data", level=2)
doc.add_paragraph(
    'The data is downloaded from Yahoo Finance, and stored in the "PreProcessedData" folder by running script "DownloadData.py". A pair of .csv files is created for each ticker, one containing exchange info, one with stock events (i.e. dividends), these file are named "[ticker name].A.csv" and "[ticker name].D.csv" respectively.'
)

doc.add_paragraph('The data in the "A" .csv file has following columns:')
doc.add_paragraph("1. Ticker: The ticker name of the asset.")
doc.add_paragraph("2. Date: The date of the trading day.")
doc.add_paragraph("3. Open: The opening price of the asset on that trading day.")
doc.add_paragraph("4. High: The highest price of the asset during that trading day.")
doc.add_paragraph("5. Low: The lowest price of the asset during that trading day.")
doc.add_paragraph("6. Close: The closing price of the asset on that trading day.")
doc.add_paragraph("7. Volume: The volume of the asset over that trading day.")
doc.add_paragraph("8. Adj Close: The adjusted closing price of the asset on that date.")


doc.add_paragraph('The data in the "D" .csv file has following columns:')
doc.add_paragraph("1. Date: The date of the trading day.")
doc.add_paragraph('2. action: The event that occurred with that asset on that date, normally just "dividend"')
doc.add_paragraph("3. value: The amount of the dividend paid per unit of the asset.")

doc.add_heading("Preprocessing Data", level=2)
doc.add_paragraph('The data ins the downloaded csv files is processed by the script "CreateTimeSeries.py", which creates a .csv file for each ticker containing the data in both the previous data frames, filling the a column with dividends with zeros for days without any events and adding a "Goal" column which represents the minimum value for which the stock traded 20 trading days after the specified Date information.')

doc.add_heading("Training Models", level=2)
doc.add_paragraph(
    'Using the ticker "'
    + ticker
    + '", which corresponds to a company that is neither too old nor too young, such that the size of the dataset is easy to handle while being possible to properly train. Furthermore, it represents an asset that has not undergone a steady value increase, which would make any strategy based on a predictive model have a hard time competing against a "Buy&Hold" naïve strategy.')
    
doc.add_paragraph(
    'A validation set is created using the 10% most recent data, while the 10% oldest data is used for test and the rest is used for training. Several models are trained and the scored are saved in a .csv file in the "Results" folder. One of the goals in this exercise was to find the smallest possible model that would yield satisfactory results for the problem at hand. While, as discussed before, the problem is intrinsically difficult and using large models would be acceptable as long as the training and activation were feasible, due to this secondary goal of the exercise, a list of possible topologies was created with the script "manipulations.py", is located in the folder "models", named "Layers_by_complexity.csv", and it specifies the amount of parameters each topology contains if it represented a standard Multi-Layer Perceptron network (MLP), despite LSTM cells being used. This is meant to provide a simple proxy to how "big" or "small" a model is. Each model has 7 neurons in an input layer and 1 in its output layer, which are often omitted when describing the topology. Topologies from the smallest to the highest complexity were tested, but were limited to 5 inner layers with at most 25 neurons each. Because of the long training times, in which many models took over 20 minutes to train, and because other operations also took considerably long, it was not possible to exhaust all the listed topologies, with most of the simplest ones being tested, and some models being skipped so it would be possible to also attempt training more complex models, which nevertheless, remained relatively simple. The aforementioned file in the "Results" folder is not overwritten if the script "PredictTicker.py" is rerun, as it checks this file for already trained models, so it is simple to fill the gaps left in the group of trained topologies should it be demanded.'
)

doc.add_paragraph('The script "PredictTicker.py" builds, compile and train models based on these topologies, then computes the scores in all data sets, saving them in a csv file with the name of the ticker in the "Results" folder, models are also saved in files located in the "models" folder, but they are not included in the repository for this file, due to space and file size limitations.')

# Load a data frame from the .csv file with the scores:
df_scores = pd.read_csv(os.path.join(results_dir, "Results_" + ticker + ".csv"))

doc.add_heading("Aggregated Results", level=2)
doc.add_paragraph("In this section, the correlation between the complexity of the models tested and the mean squared errors are shown. Because poor results were being obtained and it was not possible to test all listed models, as will be explained later, the complexity correlation plots have some clusters related to part of the models being skipped.")

PlotData.plot_complexity_correlation(df_scores, "MSE Train")
doc.add_fig()
doc.add_paragraph("As one would expect, there is a healthy negative correlation between complexity and mean squared, error, so in general, more complex models are expected and seen to be more precise for the training data at least.")

PlotData.plot_complexity_correlation(df_scores, "MSE Future")
doc.add_fig()
doc.add_paragraph("However, there is an unexpected although small positive correlation between complexity and mean squared error, which shows that the models are not generalizing well. It is fair to point out that the validation data contains some crisis events, namely the COVID-19 pandemic, and other ticker specific events which are hard to forecast, even if the 2008 financial crisis was in the training data as an example of extreme period.")

PlotData.plot_complexity_correlation(df_scores, "MSE Past")
doc.add_fig()
doc.add_paragraph("Finally, the mean squared error is seen to have negligible correlation with the test data, which is taken from the beginning of the time series.")

PlotData.plot_correlation(df_scores, "MSE Train", "MSE Future")
doc.add_fig()
doc.add_paragraph('Very unfortunately, the mean squared error in the training data has negative correlation with the future data, which is the validation set, and despite being referred as "future" is just mean the most recent portion of the time series. This implies that in the range of complexities tested, there is little generalization capacity on the trained models.')

PlotData.plot_correlation(df_scores, "MSE Train", "MSE Past")
doc.add_fig()
doc.add_paragraph('Interestingly, there is a meaningfully positive correlation between the errors in training and test data, which will partially explain some of the good results which will be presented below. This however might be due to the fact that the test data set corresponds to a period of less turbulent economic conditions, which is normally associated with less volatility and less unpredictable asset prices.')

# Load data used for training, testing and validation:
data_file = os.path.join(data_folder, ticker + ".csv")
mdl_empty = classes.ModelSpecs(data_file)
x, y, dt = mdl_empty.get_x_y(data_file)
x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = PlotData.split_sets(x, y, dt)

doc.add_heading("Sample Results", level=2)

doc.add_paragraph('In this sections, the results obtained for a few topologies are displayed, along with some plots showing how would an actual investment strategy based on this model perform on each of the sets. Normally one would expect very good results on training data, since the models were fit into this dataset, moderately good results on validation date since the best models were selected partially based on the results obtained on the validation set (which is the latest trading data) and poor results on the test data, which for this exercise is composed of the first part of the time series, which is not included in the training set.')

doc.add_paragraph('The proposed strategy consists of the following procedure: The model is run every night without retraining using all available data. This allows using data of that financial day, and create a prediction for the minimum trading price of the asset twenty financial days later. If the prediction surpasses the closing price by more than a given relative value, i.e 5%, then an investment of one monetary unit is made. Otherwise no position is taken. Because it is only possible to operate on the following financial day, the asset is assumed to be bought on the highest trading price of the consecutive day, and whatever happens, it is sold on the financial day for which the prediction was made at the lowest trading price of that day. This is a simple, if not naïve strategy based on a prediction which does have an associated time window, despite the fact that most asset forecasts avoid giving a precise and especially short term target date for a predicted value. Also, the assumption of buying on the highest trading price of the following day while selling at the lowest price of the prediction day is deliberately pessimistic, which is meant to safe-guard the strategy against poor execution of either the trader or automated system that would perform the operations. This however does not prevent large operations by this strategy to meaningfully affect the price of the asset, thus being probably a realistic scenario at small trading volumes relative to the liquidity of the asset, but an unrealistic method for very large amounts or illiquid assets.')

doc.add_paragraph('Using the aforementioned strategy, the profit is computed by assuming that exactly one financial unit was invested on each day that the relative gain criterion was satisfied. A distinction is made between the predicted profit and the feasible profit, in the sense that the predicted one is the profit that could be earned if the asset traded on the prediction date exactly at the price forecast by the model, while the feasible profit happens when the asset is sold at the value corresponding to the actual trading data. For simplicity of interpretation, the profit is added at the time of the investment, not at the time the asset would be sold. This allows visualizing how aggressively or inertly the strategy behaves by differentiating times when the strategy creates new positions and periods when it doesn\'t make any movements. To provide a relative value, a cumulative Return on Invested Capital (ROIC) chart is also plotted, and the return rate on each piece of the timeseries is computed for as an annualized return rate and mentioned on the paragraphs below, thus allowing a closer comparison between performances between datasets and topologies. It is important to notice, however, that if an asset were to always increase in price, the strategy would never lose money whatever it did. Also, any annualized return below 5% can be understood to be lower than a relatively small equity risk premium, whereas the return on a risky investment should surpass both this premium value and the risk free rate, which in Brazil, for the period between the years of 1999 and 2021 has varied between 45% and 2% a year. Also, note that these returns and profits make no assumption of compounding, the invested amount at any day has no relation with previous earnings or losses, which is unusual in traditional investing, although day traders often design strategies that require a fixed and relatively small invested capital, due to the lack of scalability of the strategy.')

doc.add_heading("First Model Results", level=3)
doc.add_paragraph(f"The first model tested has the topology (7, 1, 1), that is, only a single LSTM neuron between the input and output layers. Naturally, it was expected to train relatively well, but to generalize poorly. The reason for this assumption is that the optimization algorithm should be able to find a direct descent path more easily, although it might not yield the best results, nor encode any sophisticated logic. Additionally, some visually good results may be the outcome of simply predicting a future price as being very close to the latest prices, which while reasonable to reduce quadratic error is a completely useless prediction method to employ on an investment strategy. Apparently, something similar to this is happening as the predictions seem to be following the actual values with a delay.")
# Load model for ticker from saved pickle file:
with open(os.path.join(os.path.dirname(__file__), "models", ticker + "(7, 1, 1).pckl"), "rb") as fid:
    mdl = pickle.load(fid)
PlotData.plot_predictions(ticker, mdl)
doc.add_fig()
PlotData.plot_training(mdl)
doc.add_fig()

doc.add_heading("Strategy Returns on Test Data", level=4)
roics_ini = PlotData.strategy_gain_plot(x_ini, mdl.predict_output_ini, y_ini, dt_ini, doc, f"Past/Test Data and topology: {mdl.layers}")

doc.add_paragraph(f'The strategy when trading on the past data yielded an annualized ROICs of {100*roics_ini[0]:.2f}%, {100*roics_ini[1]:.2f}%, and {100*roics_ini[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order. This is interesting as this data set is never presented to model nor for training nor selection purposes. However, a positive result might be the outcome of the asset generally having long steady increases in value during the period, and two fast crashes. Nonetheless, these results are very underwhelming regardless of the economic conditions of the period.')

doc.add_heading("Strategy Returns on Training Data", level=4)
roics_trn = PlotData.strategy_gain_plot(x_trn, mdl.predict_output_trn, y_trn, dt_trn, doc, f"Training Data and topology: {mdl.layers}")
doc.add_paragraph(f'The strategy when trading on the training data yielded an annualized ROICs of {100*roics_trn[0]:.2f}%, {100*roics_trn[1]:.2f}%, and {100*roics_trn[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order. This is somewhat understandable because the model in this case is very small, so even if this is the training data, there is just not enough space to encode any meaningful logic in this model. Also, the Mean Squared Error metric, despite being generally useful for analytic purposes does prioritize avoiding large errors even if sparse in favor of perfectly matching most of the time with no regards to few isolated points.')

doc.add_heading("Strategy Returns on Validation Data", level=4)
roics_end = PlotData.strategy_gain_plot(x_end, mdl.predict_output_end, y_end, dt_end, doc, f"Future/Validation Data and topology: {mdl.layers}")
doc.add_paragraph(f'The strategy when trading on the most recent data yielded an annualized ROICs of {100*roics_end[0]:.2f}%, {100*roics_end[1]:.2f}%, and {100*roics_end[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order. This is partially due to the fact that for most of the concerned period the asset just fell in value, rather than increase, but these are nonetheless smaller losses than holding the asset for the whole period.')

doc.add_heading("One day of Training Results", level=3)
doc.add_paragraph('After training the simplest models for a roughly a day, the topology (7, 8, 9, 1) emerged as a best performer at that point. This section shows the results obtained for it with some comparison with the reference result obtained for the simplest model. Notably, this network did not take significantly long to train, but the time required to reach this model is mostly related to the amount of models with poor topologies that were trained in the meantime. At some point, models with hidden layers smaller than 4 neurons were not tested.')

with open(os.path.join(os.path.dirname(__file__), "models", ticker + "(7, 8, 9, 1).pckl"), "rb") as fid:
    mdl = pickle.load(fid)
PlotData.plot_predictions(ticker, mdl)
doc.add_fig()
PlotData.plot_training(mdl)
doc.add_fig()

doc.add_heading("One week of Training Results", level=3)
# find most recent file in models folder:
latest_model = max(glob.iglob(os.path.join(os.path.dirname(__file__), "models", ticker + "*.pckl")), key=os.path.getctime)

with open(latest_model, "rb") as fid:
    mdl = pickle.load(fid)

doc.add_paragraph(f'After training for more than a week, the latest model found with better score on the combination of the training and validation data was the one with the topology {mdl.layers}. It has once again not taken a significantly long time to train compared to smaller models, and neither did it present a meaningful performance improvement.')

PlotData.plot_predictions(ticker, mdl)
doc.add_fig()
PlotData.plot_training(mdl)
doc.add_fig()

doc.add_heading("Strategy Returns on Test Data", level=4)
roics_ini = PlotData.strategy_gain_plot(x_ini, mdl.predict_output_ini, y_ini, dt_ini, doc, f"Past/Test Data for topology: {mdl.layers}")
doc.add_paragraph(f'The strategy when trading on the past data yielded an annualized ROICs of {100*roics_ini[0]:.2f}%, {100*roics_ini[1]:.2f}%, and {100*roics_ini[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order.')

doc.add_heading("Strategy Returns on Training Data", level=4)
roics_trn = PlotData.strategy_gain_plot(x_trn, mdl.predict_output_trn, y_trn, dt_trn, doc, f"Training Data for topology: {mdl.layers}")
doc.add_paragraph(f'The strategy when trading on the training data yielded an annualized ROICs of {100*roics_trn[0]:.2f}%, {100*roics_trn[1]:.2f}%, and {100*roics_trn[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order.')

doc.add_heading("Strategy Returns on Validation Data", level=4)
roics_end = PlotData.strategy_gain_plot(x_end, mdl.predict_output_end, y_end, dt_end, doc, f"Future/Validation Data for topology: {mdl.layers}")
doc.add_paragraph(f'The strategy when trading on the most recent data yielded an annualized ROICs of {100*roics_end[0]:.2f}%, {100*roics_end[1]:.2f}%, and {100*roics_end[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order.')

#mdl.plot_model()

doc.add_heading("Conclusion", level=2)
doc.add_paragraph(f"The ticker {ticker} was chosen for a myriad of reasons, one of them being the fact that the lack of a consistent upward trend should prevent good results coming from checking a broken clock at the right time of the day, meaning, good results for this ticker would be unlikely to be the effect of a coincidence or the fact that the proposed strategy would always be profitable no matter the accuracy of the prediction if the asset would always appreciate over the period of twenty financial days. As explained in the introduction, the prediction problem for a financial asset is inherently difficult, and no quantitative strategy, especially one designed to be relatively simple should perform consistently over long periods of time. All that being said, the results currently obtained were significantly underwhelming, which is mostly due to the fact that as the correlation graphs show, there is little generalization capacity for the limited size of the models tested. The thresholds applied for the investment strategy do not seem to point towards there existing some sweet spot number, being too conservative when the threshold is high and though not so aggressive when the threshold is low, but nonetheless no result found for annualized return on invested capital were above a reasonable equity risk premium, let alone a proper target return rate. The exercise nonetheless allowed experimenting with time series forecasting and Long Short-Term Memory neural networks, and develop some first hand experience in implementing and training them.")


doc.finish()
# Launch the created document:
if os.name == "nt":
    os.system(f'start "" "{os.path.join(mount_dir, doc.file_name)}"')
else:
    os.system(f'xdg-open "{os.path.join(mount_dir, doc.file_name)}"')
print("Finished", os.path.basename(__file__))
