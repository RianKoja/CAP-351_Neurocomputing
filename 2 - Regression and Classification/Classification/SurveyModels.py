########################################################################################################################
# Iteratively tests different topologies for a multi-layers perceptron network.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports:
import os
import pickle

# PyPI imports:
import pandas as pd
import numpy as np

# Local imports:
from tools import classes, manipulations

# Create the number of hidden layers and neurons per layer randomly, with limitations:
def make_layers():
    this_length = np.random.randint(0, 10)  # Up to ten hidden layers
    num_list = []
    for _ in range(this_length + 1):
        num_list.append(np.random.randint(12, 900))  # Range of neurons per layer
    return tuple(num_list)


# Use fixed random seed to have well defined behavior
np.random.seed(2626354877)

# Read dataset:
dir_mount = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mount")
dir_data = os.path.join(dir_mount, "..", "dataset")
dataset_filename = os.path.join(dir_data, "winequality-merged.xlsx")
model_filename = os.path.join(dir_mount, "model_file.pkl")
model = classes.ModelSpecs(dataset_filename)

# Save untrained model:
with open(model_filename, "wb") as fid:
    pickle.dump(model, fid)

# Get the data from the dataset:
X_total, y_total = model.get_x_y(dataset_filename)

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
            "Training",
            "Validation",
            "Development",
            "Test",
            "Total",
        ],
    )
    score_max = -1e20
else:
    df_scores = pd.read_csv(csv_name, index_col="Unnamed: 0")
    score_max = max(df_scores["Development"])


while True:
    layers = make_layers()
    if not (df_scores["Layers"] == str(layers)).any():
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
        if score_dev > score_max:
            score_max = score_dev
            # check training results for other configurations of the optimization algorithm:
            backup_fit = model.fit_out
            print("Training with no momentum...")
            model.train(
                X_train,
                y_train,
                layers,
                momentum=0.0,
                nesterov=False,
                epochs=max(100, len(backup_fit.history["loss"])),
            )
            model.fit_00momentum = model.fit_out
            print("Training with 0.9 momentum (not Nesterov)...")
            model.train(
                X_train,
                y_train,
                layers,
                momentum=0.9,
                nesterov=False,
                epochs=max(100, len(backup_fit.history["loss"])),
            )
            model.fit_09momentum = model.fit_out
            model.fit_out = backup_fit
            print("Training with 1e-1 regularization...")
            model.train(
                X_train,
                y_train,
                layers,
                regularizer=1e-1,
                epochs=max(100, len(backup_fit.history["loss"])),
            )
            model.fit_reg_zero = model.fit_out
            model.fit_out = backup_fit
            print("Training with 0 regularization...")
            model.train(
                X_train,
                y_train,
                layers,
                regularizer=0,
                epochs=max(100, len(backup_fit.history["loss"])),
            )
            model.fit_reg_high = model.fit_out
            model.fit_out = backup_fit

            # Save the model with the current best score, and models with lower scores can be deleted.
            pkcl_name = "model_" + str(score_dev) + "_" + str(layers).replace(" ", "") + ".pckl"
            model_filename = os.path.join(dir_mount, pkcl_name)
            with open(model_filename, "wb") as fid:
                pickle.dump(model, fid)
        df_scores.to_csv(csv_name)

# This is expected to run until interrupted.
