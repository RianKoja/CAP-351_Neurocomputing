# Standard imports:
import copy
import os
import pickle
from scipy.stats import linregress

# PyPI imports:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# based on ticker given as input, open the data frame and plot the data, with volume in a separate subplot
def plot_series(ticker):
    processed_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "ProcessedData")
    # Open the .csv file containing the data for the ticker:
    df = pd.read_csv(os.path.join(processed_data_folder, ticker + ".csv"))
    # Set to datetime and normalize the dates:
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    # Plot the data:
    plt.figure(figsize=(16, 8))
    ax = plt.subplot(2, 1, 1)
    df.plot.bar(x="Date", y="Volume", title=ticker, ax=ax)
    # Do not show the x axis ticks:
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.legend()
    plt.title(ticker)
    # Plot the volume data as a bar chart:
    ax = plt.subplot(2, 1, 2)
    df.plot(x="Date", y=["Close", "High", "Low", "Open"], title=ticker, ax=ax)
    plt.legend()
    plt.grid(True)
    # Show only at most 20 ticks on the x axis:
    plt.setp(ax.get_xticklabels(), visible=True, rotation=45, ha="right")
    plt.draw()


# Plot predictions:
def plot_predictions(ticker, mdl):
    data_folder = os.path.join(os.path.dirname(__file__), "..", "ProcessedData")
    data_file = os.path.join(data_folder, ticker + ".csv")
    x, y, dt = mdl.get_x_y(data_file)

    x_train, y_train, dt_train, x_test_ini, y_test_ini, dt_test_ini, x_test_end, y_test_end, dt_test_end = split_sets(x, y, dt)

    plt.figure(figsize=(10, 6))
    plt.plot(mdl.fit_out.epoch, mdl.fit_out.history["loss"], label="Training")
    plt.plot(mdl.fit_out.epoch, mdl.fit_out.history["val_loss"], label="Validation")
    plt.grid("minor", "both")
    plt.title(f'Loss during training for topology: {mdl.layers}, training loss: {mdl.fit_out.history["loss"][-1]}')
    plt.legend()
    plt.tight_layout()
    plt.draw()

    # Evaluate model:
    predict_output_trn = mdl.predict_output_trn.reshape(-1)  # mdl.predict(x_train)
    predict_output_ini = mdl.predict_output_ini.reshape(-1)  # mdl.predict(x_test_ini)
    predict_output_end = mdl.predict_output_end.reshape(-1)  # mdl.predict(x_test_end)

    df_test_ini = copy.deepcopy(dt_test_ini)
    df_test_ini.insert(1, "y", y_test_ini)
    df_test_ini.insert(2, "y_pred", predict_output_ini)

    df_train = copy.deepcopy(dt_train)
    df_train.insert(1, "y", y_train)
    df_train.insert(2, "y_pred", predict_output_trn)

    df_test_end = copy.deepcopy(dt_test_end)
    df_test_end.insert(1, "y", y_test_end)
    df_test_end.insert(2, "y_pred", predict_output_end)

    # Plot model results against test data:
    ax = df_test_end.plot(x="Date", y="y", label="y_test_end", figsize=(10, 6), linestyle='dashed', color='xkcd:blue')
    ax = df_test_end.plot(x="Date", y="y_pred", label="y_pred_end", ax=ax, color='xkcd:royal blue')
    df_train.plot(x="Date", y="y", label="y_train", ax=ax, linestyle='dashed', color='xkcd:green')
    df_train.plot(x="Date", y="y_pred", label="y_pred_train", ax=ax, color='xkcd:forest green')
    df_test_ini.plot(x="Date", y="y", label="y_test_ini", ax=ax, linestyle='dashed', color='xkcd:red')
    df_test_ini.plot(x="Date", y="y_pred", label="y_pred_ini", ax=ax, color='xkcd:dark red')
    plt.title(f'Predictions vs reference for topology: {mdl.layers}, training loss: {mdl.fit_out.history["loss"][-1]:.4f}, validation loss:{mdl.fit_out.history["val_loss"][-1]:.4f}')
    plt.grid("minor", "both")
    plt.tight_layout()
    plt.draw()


# Plot training:
def plot_training(mdl):
    plt.figure(figsize=(10, 6))
    plt.plot(mdl.fit_out.epoch, mdl.fit_out.history["loss"], label="Training")
    plt.plot(mdl.fit_out.epoch, mdl.fit_out.history["val_loss"], label="Validation")
    plt.grid("minor", "both")
    plt.title(f'Loss during training for topology: {mdl.layers}, Complexity: {mdl.complexity}, training loss: {mdl.fit_out.history["loss"][-1]:.4f}, validation loss:{mdl.fit_out.history["val_loss"][-1]:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.draw()


def split_sets(x, y, dt):
    # Separate train and test data:
    split_ini = int(len(x) * 0.15)
    split_end = int(len(x) * 0.85)
    x_train = x[split_ini:split_end]
    y_train = y[split_ini:split_end]
    dt_train = dt.iloc[split_ini:split_end]
    x_test_ini = x[:split_ini]
    y_test_ini = y[:split_ini]
    dt_test_ini = dt.iloc[:split_ini]
    x_test_end = x[split_end:]
    y_test_end = y[split_end:]
    dt_test_end = dt.iloc[split_end:]

    return x_train, y_train, dt_train, x_test_ini, y_test_ini, dt_test_ini, x_test_end, y_test_end, dt_test_end


# Creates a scatter plot with the regression between MSE and complexity:
def plot_complexity_correlation(df_scores, col_name):
    # Compute linear regression between 'Complexity' and 'MSE Train':
    slope, intercept, r_value, p_value, std_err = linregress(df_scores["Complexity"], df_scores[col_name])

    # Create a scatter plot of the scores and plot the linear regression:
    ax = df_scores.plot.scatter("Complexity", col_name, label="Data")
    ax.plot(df_scores["Complexity"], intercept + slope * df_scores["Complexity"], "-", color="red", label="Regression Line")
    # Add labels:
    ax.set_title("%s vs Complexity\n correlation = %.3f" % (col_name, r_value))
    ax.set_xlabel("Complexity")
    ax.set_ylabel(col_name)
    plt.draw()


# Creates a plot comparing maximum prices on the day after the prediction and the predicted price:
def compare_prediction(x, y_pred, dt, doc):
    # Extract only the "maximum price" (i.e. first) column in x:
    y_max = x[:, 0]
    # remove first row:
    y_max = y_max[1:]
    # Remove last row of y_pred and dt:
    y_pred = y_pred[:-1]
    dt = dt[:-1]
    # Add columns to df:
    df_compare = copy.deepcopy(dt)
    df_compare["y_max"] = y_max
    df_compare["y_pred"] = y_pred.flatten()  # $y_pred.tolist() #.reshape(-1)
    # Plot the two with dt on x-axis:
    df_compare.plot(x="Date", y="y_max", label="max price", color='b', linestyle='dashed')
    df_compare.plot(x="Date", y="y_pred", label="prediction", color='b')
    plt.title("Prediction for 20 days later vs maximum prices on following trading day")
    plt.grid("minor", "both")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    doc.add_fig()
    # Plot the difference of the two:
    df_compare["diff"] = df_compare["y_pred"] - df_compare["y_max"]
    df_compare.plot(x="Date", y=["diff"], label=["Difference"])
    plt.title("Difference between Prediction for 20 days later\nvs maximum prices on following trading day")
    plt.grid("minor", "both")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    doc.add_fig()


def strategy_gain_plot(x, y_pred, y_real, dt, doc, data_name, threshold_hig=0.05, threshold_med=0.03, threshold_low=0.01):
    # Assume that each time the forecast is above the closing price by about the threshold value, a purchase of 1
    # monetary unit is made on the asset. Plot the investment made, the current balance and the return.
    fig_profit = plt.figure(figsize=(10, 6))
    fig_roic = plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']

    for threshold, color in zip([threshold_hig, threshold_med, threshold_low], colors):
        daily_investment = []
        daily_predicted_return = []
        daily_real_return = []

        for ii in range(0, len(y_pred)-1):
            price_closing_today = x[ii, 3]
            prediction = y_pred[ii]
            future_value = y_real[ii][0]
            purchase_price = x[ii+1, 0]
            predicted_return = y_pred[ii]/price_closing_today - 1
            if predicted_return > threshold:  # Make investment
                daily_investment.append(1)
                daily_predicted_return.append(predicted_return)
                price_high_tomorrow = x[ii+1, 0]
                daily_real_return.append(y_real[ii][0]/price_high_tomorrow - 1)
            else:
                daily_investment.append(0)
                daily_predicted_return.append(0)
                daily_real_return.append(0)

        df = dt[:-1].copy()
        df["Cumulative Investment"] = np.cumsum(daily_investment)
        df["Cumulative Predicted Profit"] = np.cumsum(daily_predicted_return)
        df["Cumulative Feasible Profit"] = np.cumsum(daily_real_return)

        df['Investment frequency'] = np.cumsum(daily_investment)/np.linspace(1, len(daily_investment), len(daily_investment), dtype=np.float64)
        df['ROIC'] = df["Cumulative Feasible Profit"]/df["Cumulative Investment"]

        plt.figure(fig_profit.number)
        ax = plt.gca()
        df.plot(x="Date", y="Cumulative Predicted Profit", label=f"Cumulative Predicted Profit @{threshold}", ax=ax, color=color, linestyle='dashed')
        df.plot(x="Date", y="Cumulative Feasible Profit", label=f"Cumulative Feasible Profit @{threshold}", ax=ax, color=color)

        plt.figure(fig_roic.number)
        ax = plt.gca()
        df.plot(x="Date", y=['ROIC'], label=[f'ROIC@{threshold}'], ax=ax)

    plt.figure(fig_profit.number)
    plt.title(f"Simulated Strategy performance at different thresholds for {data_name}")
    plt.grid("minor", "both")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    doc.add_fig()

    plt.figure(fig_roic.number)
    plt.title(f"Simulated Strategy ROIC at different thresholds for {data_name}")
    plt.grid("minor", "both")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    doc.add_fig()


if __name__ == "__main__":
    plot_series("BOVA11")

    plot_series("ITSA4")

    plt.show()

    print("Finished", __file__)
