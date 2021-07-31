########################################################################################################################
# Iteratively tests different topologies for a multi-layers perceptron network.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

import os
import pickle
import sys

# PyPI imports:
from numpy.core.numeric import Inf
import pandas as pd
import numpy as np

# Local imports:
from tools import classes, manipulations


# Create the number of hidden layers and neurons per layer randomly, with limitations:
def make_layers():
    this_length = np.random.randint(0, 10)  # Up to ten hidden layers
    num_list = []
    for _ in range(this_length + 1):
        num_list.append(np.random.randint(20, 800))  # Range of neurons per layer
    return tuple(num_list)


# Check if any input was given, assume it is the maximum number of runs if so:
try:
    max_runs = int(sys.argv[1])
except:
    max_runs = Inf

# Read dataset:
dir_mount = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mount")
dataset_xlsx_filename = os.path.join(dir_mount, "boston_dataset.xlsx")
model_filename = os.path.join(dir_mount, "model_file.pkl")
model = classes.ModelSpecs(dataset_xlsx_filename)

# Save untrained model:
with open(model_filename, "wb") as fid:
    pickle.dump(model, fid)

# Load the dataset:
X_total, y_total = model.get_x_y(dataset_xlsx_filename)

# Break into training, validation and test data:
(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_dev,
    y_dev,
    X_tests,
    y_tests,
) = manipulations.split_data(model, X_total, y_total)

# Create Log:
csv_name = os.path.join(dir_mount, "Results.csv")
if not os.path.isfile(csv_name):
    df_scores = pd.DataFrame(
        [],
        columns=[
            "Layers",
            "Score Train",
            "Score Validation",
            "Score Dev",
            "Score Test",
            "Score Total",
        ],
    )
    score_max = -1e20
else:
    df_scores = pd.read_csv(csv_name, index_col="Unnamed: 0")
    score_max = max(df_scores["Score Dev"])

# Use fixed random seed to have well defined behavior
np.random.seed(262635487)

iterations = 0
while iterations < max_runs:
    layers = make_layers()
    if not (df_scores["Layers"] == str(layers)).any():
        iterations += 1
        print("testing for hidden layers =", layers)
        model.train(X_train, y_train, layers)
        score_train = model.score(X_train, y_train)
        score_valid = model.score(X_valid, y_valid)
        score_dev = model.score(X_dev, y_dev)
        score_test = model.score(X_tests, y_tests)
        score_total = model.score(X_total, y_total)
        df_scores.loc[len(df_scores)] = [
            layers,
            score_train,
            score_valid,
            score_dev,
            score_test,
            score_total,
        ]
        # For models with same score, comparison is performed on file "CompareModels.py"
        if score_dev >= score_max:
            score_max = score_dev
            # Add additional information to the model for reporting purposes:
            # first backup some data:
            loss_curve_original = model.clf.loss_curve_
            loss_original = model.clf.loss_
            coefs_original = model.clf.coefs_
            intercepts_original = model.clf.intercepts_
            print("Training with no momentum...")
            model.train(
                X_train,
                y_train,
                layers,
                nesterovs_momentum=False,
                momentum=0.0,
                max_iter=max(100, len(loss_curve_original)),
            )
            model.loss_curve_zeromom = model.clf.loss_curve_

            print("Training with 0.9 momentum (not Nesterov)...")
            model.train(
                X_train,
                y_train,
                layers,
                nesterovs_momentum=False,
                momentum=0.9,
                max_iter=max(100, len(loss_curve_original)),
            )
            model.loss_curve_09mom = model.clf.loss_curve_
            print("Training with 1e-1 regularization...")
            model.train(
                X_train,
                y_train,
                layers,
                alpha_regularization=1e-1,
                max_iter=max(100, len(loss_curve_original)),
            )
            model.loss_reg_zero = model.clf.loss_curve_
            print("Training with 0 regularization...")
            model.train(
                X_train,
                y_train,
                layers,
                alpha_regularization=0,
                max_iter=max(100, len(loss_curve_original)),
            )

            model.loss_reg_high = model.clf.loss_curve_

            # Recover attributes from original training:
            model.clf.loss_curve_ = loss_curve_original
            model.clf.loss_ = loss_original
            model.clf.coefs_ = coefs_original
            model.clf.intercepts_ = intercepts_original
            # Save the model with the current best score, and models with lower scores can be deleted.
            pkcl_name = "model_" + str(score_dev) + "_" + str(layers).replace(" ", "") + ".pckl"
            model_filename = os.path.join(dir_mount, pkcl_name)
            with open(model_filename, "wb") as fid:
                pickle.dump(model, fid)
        df_scores.to_csv(csv_name)

# This is expected to run until interrupted if no argument was given on input.
