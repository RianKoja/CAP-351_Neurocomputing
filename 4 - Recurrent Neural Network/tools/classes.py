########################################################################################################################
# Create a class that allows saving and loading a configured and trained neural network model.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports
import sys
import os
import time

# PyPI imports
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds." % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


# Make keras models pickleable:
def __getstate__(self):
    model_str = ""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fd:
        tf.keras.models.save_model(self, fd.name, overwrite=True)
        model_str = fd.read()
    d = {"model_str": model_str}
    return d


def __setstate__(self, state):
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fd:
        fd.write(state["model_str"])
        fd.flush()
        model = tf.keras.models.load_model(fd.name)
    self.__dict__ = model.__dict__


class ModelSpecs:
    def __init__(self, data_file):
        # Prepare keras:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ref: https://stackoverflow.com/a/42121886/3007075
        # tf.get_logger().setLevel('ERROR')
        tf.keras.backend.clear_session()
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Read files:
        x, y, _ = self.get_x_y(data_file)

        # Determine sizes:
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.dim_output = y.shape[1]

        # Create scalers:
        self.scaler_x = StandardScaler()
        self.scaler_x.fit(x)
        self.scaler_y = StandardScaler()
        self.scaler_y.fit(y)

        # Add a predictor model and make it pickleable:
        self.rnn = None

        # Add some fields for later use:
        self.layers = None
        self.complexity = None
        self.fit_out = None

        # Because models retrieved from pickle file fail to make predictions, save the prediction result:
        self.predict_output_trn = None
        self.predict_output_ini = None
        self.predict_output_end = None

    # Provide input and output from csv files:
    def get_x_y(self, file):
        df = None
        if file.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.endswith(".xlsx"):
            df = pd.read_excel(file)
        if df is None:
            print("Data file not found!")
            exit(3)
        df_x = df.drop(columns=["Date", "Goal"], axis=1, inplace=False)
        x = np.array(df_x.values.tolist(), dtype=np.float64)
        y = np.array(df["Goal"].tolist(), dtype=np.float64).ravel().reshape(-1, 1)
        # Provide datetime column:
        dt = pd.DataFrame()
        dt["Date"] = pd.to_datetime(df["Date"])
        return x, y, dt

    @timeit
    def train(self, x_train, y_train, x_val, y_val, layers, epochs=20000, batch_size=30):
        self.rnn = tf.keras.models.Sequential()
        self.rnn.__getstate__ = lambda mdl=self.rnn: __getstate__(mdl)
        self.rnn.__setstate__ = __setstate__
        self.layers = layers
        self.build()
        # reshape from into [timesteps, batch, features]
        x_scaled = np.reshape(self.scaler_x.transform(x_train), (x_train.shape[0], 1, x_train.shape[1]))
        y_scaled = np.reshape(self.scaler_y.transform(y_train), (y_train.shape[0], 1, y_train.shape[1]))
        x_val_scaled = np.reshape(self.scaler_x.transform(x_val), (x_val.shape[0], 1, x_val.shape[1]))
        y_val_scaled = np.reshape(self.scaler_y.transform(y_val), (y_val.shape[0], 1, y_val.shape[1]))
        # Split data into batch_size batches to speed-up training, but ditch last datapoints if needed to ensure divisibility:
        if batch_size == 1:
            x_split = x_scaled
            y_split = y_scaled
        else:
            x_split = np.concatenate(np.array_split(x_scaled[: -(x_train.shape[0] % batch_size)], batch_size), axis=1)
            y_split = np.concatenate(np.array_split(y_scaled[: -(x_train.shape[0] % batch_size)], batch_size), axis=1)
        callback = [tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=500)]  # , restore_best_weights=True)]
        self.fit_out = self.rnn.fit(x_split, y_split, epochs=epochs, callbacks=callback, validation_data=(x_val_scaled, y_val_scaled), verbose=0)

    def score(self, x, y):
        x_scaled = np.reshape(self.scaler_x.transform(x), (x.shape[0], 1, x.shape[1]))
        y_scaled = np.reshape(self.scaler_y.transform(y), (y.shape[0], 1, y.shape[1]))
        result = self.rnn.evaluate(x_scaled, y_scaled, verbose=1, batch_size=1, return_dict=True)
        # print('dict:', result)
        # print('self.rnn.metrics_names = ', self.rnn.metrics_names)
        return result["loss"]

    def predict(self, x):
        x_scaled = np.reshape(self.scaler_x.transform(x), (x.shape[0], 1, x.shape[1]))
        predict_output = self.rnn(x_scaled)
        return self.scaler_y.inverse_transform(np.reshape(predict_output, -1))

    def build(self):
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
        # Only the last hidden layer may be set with "return_sequences=False"... Bu I want the whole sequence in output for plotting graphs
        self.rnn.add(tf.keras.layers.LSTM(hidden_layers[-1], activation="tanh", return_sequences=True, unroll=False,
                                          recurrent_activation="sigmoid", use_bias=True, time_major=True,
                                          recurrent_dropout=0, stateful=False, input_shape=(None, 1, int(last_layer))))

        # Create output layer:
        self.rnn.add(tf.keras.layers.Dense(self.dim_output, activation="linear"))
        self.compile()

    def compile(self):
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
        self.rnn.compile(loss="mse", optimizer=adam_optimizer)

    def plot_model(self):
        tf.keras.utils.plot_model(self.rnn, show_shapes=True, to_file=f"{self.layers}.png", dpi=300)
