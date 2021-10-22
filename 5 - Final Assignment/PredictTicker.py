########################################################################################################################
# Train model for predicting a specified stock ticker's low price 20 financial days in the future days based on the
# sequence of days until given date. Uses the model in the ModelSpecs class. Evaluates different topologies searching
# for the smallest viable one.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports:
import copy
import os
import pickle
import time

# PyPI imports:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Local imports:
from tools import classes, manipulations


print("Starting", os.path.basename(__file__))

config = classes.Config()

# Timesteps per batch:
t_steps_per_batch = 252  # One year of trading days, approximately

# Prepare keras:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ref: https://stackoverflow.com/a/42121886/3007075
# tf.get_logger().setLevel('ERROR')
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Relevant paths:
data_folder = os.path.join(os.path.dirname(__file__), "ProcessedData")
models_folder = os.path.join(os.path.dirname(__file__), "Models")
results_folder = os.path.join(os.path.dirname(__file__), "Results")
figs_folder = os.path.join(os.path.dirname(__file__), "LaTeX_report", 'figs')

for folder in [models_folder, results_folder, figs_folder]:
    os.makedirs(folder, exist_ok=True)

# Create empty model:
mdl_empty = classes.ModelSpecs()

# Get data
x_trn, y_trn, x_val, y_val = None, None, None, None
for ticker in config.target_tickers:
    x, y, dt = mdl_empty.get_x_y(data_folder, ticker, config.reference_tickers)
    x_trn_ticker, y_trn_ticker, _, _, _, _, x_end_ticker, y_end_ticker, _ = manipulations.split_sets(x, y, dt)
    dividend, module = divmod(x_trn_ticker.shape[0], t_steps_per_batch)
    if dividend < 1:
        raise Exception(f"Ticker {ticker} does not have sufficient history time for building a training set")
    x_split = np.concatenate(np.array_split(x_trn_ticker[: -module], dividend, axis=0), axis=1)
    y_split = np.concatenate(np.array_split(y_trn_ticker[: -module], dividend, axis=0), axis=1)
    dividend, module = divmod(x_end_ticker.shape[0], t_steps_per_batch)
    if dividend < 1:
        raise Exception(f"Ticker {ticker} does not have sufficient history time for building a validation set")
    x_val_split = np.concatenate(np.array_split(x_end_ticker[: -module], dividend, axis=0), axis=1)
    y_val_split = np.concatenate(np.array_split(y_end_ticker[: -module], dividend, axis=0), axis=1)
    if x_trn is None:
        x_trn = x_split
        y_trn = y_split
        x_val = x_val_split
        y_val = y_val_split
    else:
        x_trn = np.concatenate((x_trn, x_split), axis=1)
        y_trn = np.concatenate((y_trn, y_split), axis=1)
        x_val = np.concatenate((x_val, x_val_split), axis=1)
        y_val = np.concatenate((y_val, y_val_split), axis=1)

# Free unused variables to reduce memory usage:
del x, y, dt

# A list of possible topologies, ordered by complexity (number of parameters if MLP) is created with tools.manipulations.py:
layers_file = os.path.join(os.path.dirname(__file__), "Layers_to_test.csv")
layers_df = pd.read_csv(layers_file)

# Use as "True" for development purposes:
plotting = False

# Create Log:
csv_name = os.path.join(results_folder, "Results_" + ".csv")
if os.path.isfile(csv_name):
    df_scores = pd.read_csv(csv_name, index_col=None)
else:
    df_scores = pd.DataFrame([], columns=["Layers", "MSE Train", "MSE Future", "MSE Train and Test"], index=None)
    df_scores.to_csv(csv_name, index=False)

if df_scores.empty:
    mse_min = np.inf
else:
    mse_min = min(df_scores["MSE Train and Test"])

# Use this to make order random:
# layers_df = layers_df.sample(frac=1)

for index, row in layers_df.iterrows():
    layers_str = row["Layers"]
    if layers_str not in df_scores["Layers"].to_list():
        layers = tuple(map(int, layers_str[1:-1].split(",")))
        hidden_layers = layers[1:-1]
        print(f"Working on layers = {layers}")

        mdl = classes.ModelSpecs()
        mdl.n_features = x_trn.shape[2]
        mdl.build(layers)
        mdl.train(x_trn, y_trn, x_val, y_val)

        if plotting:
            plt.figure(figsize=(10, 6))
            plt.plot(mdl.fit_history["loss"], label="Training")
            plt.plot(mdl.fit_history["val_loss"], label="Validation")
            plt.grid("minor", "both")
            plt.title(f'Loss during training for topology: {layers_str}, training loss: {mdl.fit_history["loss"][-1]}')
            plt.legend()
            plt.tight_layout()
            plt.draw()

        # Use this arithmetic mean to have a composite loss value:
        mse_train_and_test = (mdl.fit_history['loss'][-1] * x_trn.shape[1] + mdl.fit_history['val_loss'][-1] * x_val.shape[1])/(x_trn.shape[1] + x_val.shape[1])
        # Save model with pickle if best accuracy found so far:
        if mse_min > mse_train_and_test:
            mse_min = mse_train_and_test
        model_filename = os.path.join(models_folder, "Model" + layers_str + ".pckl")
        mdl.save_rnn(models_folder)
        with open(model_filename, "wb") as fid:
            pickle.dump(mdl, fid)
        mdl.load_rnn(models_folder)

        # Re-read the file before appending new result so multiple instances of this script can run in parallel:
        df_scores = pd.read_csv(csv_name, index_col=None)
        df_scores.loc[len(df_scores)] = [layers_str, mdl.score(x_trn, y_trn), mdl.score(x_val, y_val), mse_train_and_test]
        df_scores.to_csv(csv_name, index=False)

        mdl.plot_model(figs_folder)
        if plotting:
            os.system(f'start "" "{figs_folder}{os.sep}{mdl.layers}.png"')

if plotting:
    print("Before plt.show, close graphs to finish program.")
    plt.show()

print("Finished", os.path.basename(__file__))
