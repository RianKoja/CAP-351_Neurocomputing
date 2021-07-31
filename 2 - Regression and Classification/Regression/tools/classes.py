########################################################################################################################
# Provide a class that saved and handles an Multi-layer Perceptron Network for regression.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# PyPI imports:
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ModelSpecs:
    def __init__(self, xlsx_filename=None):
        self.n_samples = None
        self.n_features = None
        self.x, self.y = self.get_x_y(xlsx_filename)

        # Create scalers:
        self.scaler_inp = StandardScaler()
        self.scaler_inp.fit(self.x)
        self.scaler_out = MinMaxScaler()
        self.scaler_out.fit(self.y.reshape(-1, 1))

        # Leave space for classifier:
        self.clf = None
        self.loss_curve_zeromom = None
        self.loss_reg_zero = None
        self.loss_reg_high = None

    def train(
        self,
        x_train,
        y_train,
        layers,
        nesterovs_momentum=True,
        momentum=0.9,
        max_iter=8000,
        alpha_regularization=1e-4,
    ):
        self.clf = MLPRegressor(
            solver="sgd",
            alpha=alpha_regularization,
            hidden_layer_sizes=layers,
            random_state=1,
            tol=1e-6,
            max_iter=max_iter,
            activation="relu",
            early_stopping=False,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
        )
        x_scaled, y_scaled = self.scale_x_y(x_train, y_train)
        self.clf.fit(x_scaled, y_scaled.ravel())

    def scale_x_y(self, x, y):
        x_scaled = self.scaler_inp.transform(x)
        y_scaled = self.scaler_out.transform(y.reshape(-1, 1))
        return x_scaled, y_scaled

    def score(self, x, y, sample_weight=None):
        x_scaled, y_scaled = self.scale_x_y(x, y)
        return self.clf.score(x_scaled, y_scaled, sample_weight=sample_weight)

    def predict(self, x):
        x_scaled = self.scaler_inp.transform(x)
        y_pred_scaled = self.clf.predict(x_scaled)
        y_pred = self.scaler_out.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred

    def get_x_y(self, xlsx_filename):
        # Read excel file:
        df = pd.read_excel(xlsx_filename)
        # Separate the column with the value to be predicted (Rent Amount)
        y = np.array(df["target"].values.tolist(), dtype=np.float64)
        df.drop(labels="target", axis="columns", inplace=True)
        # Assing the input values to the x variable:
        x = np.array(df.values.tolist(), dtype=np.float64)

        # Determine sizes:
        if self.n_samples is None:
            self.n_samples = len(y)
            self.n_features = len(x[0])

        return x, y

    def predict_from_xlsx(self, xlsx_filename):
        x, _ = self.get_x_y(xlsx_filename)
        x_scaled = self.scaler_inp.transform(x)
        y_pred = self.predict(x_scaled)
        return y_pred
