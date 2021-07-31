########################################################################################################################
# Create a class that allows saving and loading a configured and trained neural network model.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports
import sys


# PyPI imports
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import keras
import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf


# Make keras models pickleable:
import types
import tempfile
import keras.models


def __getstate__(self):
    model_str = ""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fd:
        keras.models.save_model(self, fd.name, overwrite=True)
        model_str = fd.read()
    d = {"model_str": model_str}
    return d


def __setstate__(self, state):
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as fd:
        fd.write(state["model_str"])
        fd.flush()
        model = keras.models.load_model(fd.name)
    self.__dict__ = model.__dict__


class ModelSpecs:
    def __init__(self, data_file):
        # Prepare keras:
        keras.backend.clear_session()
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Read files:
        x, y = self.get_x_y(data_file, scale=False)

        # Determine sizes:
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]

        # Create scaler:
        self.scaler_x = StandardScaler()
        self.scaler_x.fit(x)

        # List existing classes:
        self.classes_list = np.unique(y)
        self.n_classes = len(self.classes_list)

        # Add a classifier and make it pickleable:
        self.classifier = keras.models.Model
        self.classifier.__getstate__ = __getstate__
        self.classifier.__setstate__ = __setstate__

        # Add some fiels for later use:
        self.fit_out = None
        self.fit_00momentum = None
        self.fit_09momentum = None
        self.fit_reg_zero = None
        self.fit_reg_high = None

    # Provide input and output from csv files:
    def get_x_y(self, file, scale=True):
        if file.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.endswith(".xlsx"):
            df = pd.read_excel(file)
        df_x = df.drop(columns=["target"], axis=1, inplace=False)
        x = np.array(df_x.values.tolist(), dtype=np.float64)
        y = np.array(df["target"].tolist(), dtype=np.int).ravel()
        if scale:
            return self.scaler_x.transform(x), self.scaler_y(y)
        else:
            return x, y

    def scaler_y(self, y):
        return np_utils.to_categorical(y, self.n_classes, dtype=np.int)

    def train(
        self,
        x_train,
        y_train,
        layers,
        momentum=0.9,
        nesterov=True,
        regularizer=1e-4,
        epochs=8000,
    ):
        self.layers = layers
        self.build(momentum, nesterov, regularizer)
        self.fit_out = self.classifier.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=8000,
            verbose=0,
            validation_split=0,
        )

    def score(self, x, y):
        result = self.classifier.evaluate(x, y, verbose=1, batch_size=1000)
        print(f"Test results - Loss: {result[0]} - Accuracy: {result[1]}%")
        ret_val = result[1]
        return ret_val

    def predict(self, x):
        predict_output = self.classifier.predict(x)
        probability = np.sum(np.max(predict_output, axis=1))
        return predict_output, probability

    def probability(self, x):
        probability = np.sum(np.max(self.classifier.predict(x), axis=1))
        return probability

    def build(self, momentum, nesterov, regularizer):
        layers = list(self.layers)
        tf.keras.regularizers.L2(l2=regularizer)
        self.classifier = keras.models.Sequential()
        # Add first layer:
        self.classifier.add(keras.layers.Dense(layers.pop(), input_shape=(self.n_features,), activation="relu"))
        # Add other layers:
        for layer in layers:
            self.classifier.add(keras.layers.Dense(layer, activation="relu"))
        # Create output layer:
        self.classifier.add(keras.layers.Dense(self.n_classes, activation="softmax"))
        self.compile(momentum, nesterov)

    def compile(self, momentum, nesterov):
        sgd_optimizer = tf.keras.optimizers.SGD(
            momentum=momentum,
            nesterov=nesterov,
            name="SGD",
        )
        self.classifier.compile(
            loss="categorical_crossentropy",
            optimizer=sgd_optimizer,
            metrics=["accuracy"],
        )

    def plot_model(self):
        tf.keras.utils.plot_model(self.classifier, show_shapes=True)
