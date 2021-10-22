########################################################################################################################
# Create a class that allows saving and loading a configured and trained neural network model.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports
import copy
import os
import sys
import time

# PyPI imports
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf


# Use this class to access parameters that will be accessed in different files:
class Config:
    def __init__(self):
        # Tickers to use in training:
        self.target_tickers = sorted(["BBAS3", "EMBR3", "MGLU3", "ENGI4", 'ELET6', 'ITUB3', 'CIEL3', "JBSS3", "UGPA3",
                                      "GGBR4", "VALE3", "SBSP3", "PSSA3", "ABEV3", "USIM5", "LREN3", "RENT3", "CCRO3",
                                      "AMER3", "WEGE3", "BRKM5", "MRFG3", "CSAN3", "BRFS3", "CVCB3",
                                      "RADL3", "TOTS3", "CYRE3", "ALPA4", "BBSE3", "VIVT3", "SANB11"])
        # Reference tickers to be used:
        self.reference_tickers = ["BOVA11", "BRAX11", "IVVB11", "KNCR11", "KNRI11"]


# Decorator for measuring function's running time.
def timeit(func):
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds." % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


class ModelSpecs:
    def __init__(self):
        # Determine sizes:
        self.n_features = None
        self.dim_output = 1

        # Add an attribute for a predictor model:
        self.rnn = None

        # Add some fields for later use:
        self.layers = None
        self.topology = None
        self.fit_history = None

    # Provide input and output from csv files:
    @staticmethod
    def get_x_y(data_folder, target_ticker, reference_tickers, dropna=True):
        df_ref = pd.DataFrame()
        for ticker in reference_tickers:
            data_file = os.path.join(data_folder, ticker + ".csv")
            df = pd.read_csv(data_file)
            df.drop('Goal', axis=1, inplace=True)
            # Append ticker name to names of columns:
            df.columns = ["Date"] + [ticker + "_" + col for col in df.columns if col != "Date"]
            # Check if df_ref is empty:
            if df_ref.empty:  # If so, then just copy:
                df_ref = df.copy()
            else:  # Append to df_ref:
                df_ref = df_ref.merge(df, on="Date", how="left", copy=True)
        # Remove columns with empty entries:
        if dropna is True:
            df_ref.dropna(inplace=True)

        # No read info on the target ticker:
        data_file = os.path.join(data_folder, target_ticker + ".csv")
        df = pd.read_csv(data_file)
        df = df_ref.merge(df, on="Date", how="left", copy=True)
        df.dropna(inplace=True)
        df_x = df.drop(columns=["Date", "Goal"], axis=1, inplace=False)
        x_array = np.array(df_x.values, dtype=np.float64)
        y_array = np.array(df["Goal"].tolist(), dtype=np.float64).ravel().reshape(-1, 1)
        # reshape form into [timesteps, batch, features]
        x = np.reshape(x_array, (x_array.shape[0], 1, x_array.shape[1]))
        y = np.reshape(y_array, (y_array.shape[0], 1, y_array.shape[1]))

        # Provide datetime column:
        dt = pd.DataFrame()
        dt["Date"] = pd.to_datetime(df["Date"])

        return x, y, dt

    @timeit
    def train(self, x_train, y_train, x_val, y_val, epochs=20000):
        callback = [tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=500)]
        fit_out = self.rnn.fit(x_train, y_train, epochs=epochs, callbacks=callback, validation_data=(x_val, y_val),
                               verbose=0)
        self.fit_history = copy.deepcopy(fit_out.history)

    def score(self, x, y):
        result = self.rnn.evaluate(x, y, return_dict=True)
        return result["loss"]

    def predict(self, x):
        return self.rnn(x)

    def build(self, layers):
        self.rnn = tf.keras.models.Sequential()
        self.layers = layers
        self.topology = f"{self.layers[0]},  {len(self.layers)-2}x{self.layers[1]}, {self.layers[-1]}"
        hidden_layers = list(self.layers[1:-1])
        # Add first layer:
        self.rnn.add(tf.keras.layers.Input(shape=(1, self.n_features)))
        # Add other hidden_layers:
        last_layer = self.n_features
        for layer in hidden_layers[:-1]:
            # input_shape=(timesteps, input_dim)
            self.rnn.add(tf.keras.layers.LSTM(layer, activation="tanh", return_sequences=True, unroll=False,
                                              recurrent_activation="sigmoid", use_bias=True, time_major=True,
                                              recurrent_dropout=0, stateful=False, input_shape=(None, 1, int(last_layer))))
            last_layer = layer
        # Only the last hidden layer may be set with "return_sequences=False"... But I want the whole sequence in output for plotting graphs
        self.rnn.add(tf.keras.layers.LSTM(hidden_layers[-1], activation="tanh", return_sequences=True, unroll=False,
                                          recurrent_activation="sigmoid", use_bias=True, time_major=True,
                                          recurrent_dropout=0, stateful=False, input_shape=(None, 1, int(last_layer))))

        # Create output layer:
        self.rnn.add(tf.keras.layers.Dense(self.dim_output, activation="linear"))
        self.compile()

    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-06, amsgrad=False, name="Adam")
        self.rnn.compile(loss="mse", optimizer=optimizer)

    def plot_model(self, save_path):
        file_name = os.path.join(save_path, f"{self.topology}.png".replace(" ", ""))
        try:
            tf.keras.utils.plot_model(self.rnn, show_shapes=True, to_file=file_name, dpi=300)
        except OSError:
            file_name = os.path.join(save_path, f"{np.random.randint(0, 999999999)}.png")
            tf.keras.utils.plot_model(self.rnn, show_shapes=True, to_file=file_name, dpi=300)

    def save_rnn(self, save_path):
        try:
            with open(os.path.join(save_path, f"rnn_{self.topology}.hdf5".replace(" ", "")), 'wb') as fid:
                tf.keras.models.save_model(self.rnn, fid.name, overwrite=True)
        except OSError:
            with open(os.path.join(save_path, f"rnn_randname_{np.random.randint(0, 999999999)}.hdf5"), 'wb') as fid:
                tf.keras.models.save_model(self.rnn, fid.name, overwrite=True)

        self.rnn = None

    def load_rnn(self, save_path):
        if self.rnn is not None:
            raise Exception("Should not overwrite model.")
        filename = os.path.join(save_path, f"rnn_{self.topology}.hdf5".replace(" ", ""))
        self.rnn = tf.keras.models.load_model(filename)
