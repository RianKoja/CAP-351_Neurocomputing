########################################################################################################################
# Run a saved model, use latest pickle file of file given at input. Call in console with -h flag for further info.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports:
import glob
import os
import pickle
import argparse
import re

# PyPI imports:
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from tools.classes import ModelSpecs  # This is opened with pickle
from tools.print_table import render_mpl_table
from tools.manipulations import load_pickle


if __name__ == "__main__":

    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        action="store",
        dest="model_picklefile",
        help="File containing pickled model to be used",
        default=None,
    )
    inputs = parser.parse_args()

    model, layers = load_pickle(inputs.model_picklefile)

    sgd_loss_curve = model.clf.loss_curve_
    print("################################")
    # add a figure with the training history
    plt.figure()
    plt.plot(model.clf.loss_curve_)
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    print("################################")
    print("################################")
    exit(33)

    # Read dataset:
    dir_mount = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mount")
    dataset_xlsx_filename = os.path.join(dir_mount, "boston_dataset.xlsx")
    # Assign to variables used by the model:
    X_total, y_total = model.get_x_y(dataset_xlsx_filename)

    # Print predictions vs. true values in a csv file:
    predictions_df = pd.DataFrame(model.predict(X_total), columns=["Predictions"])
    predictions_df["True Values"] = y_total
    # Add error column:
    predictions_df["Error"] = predictions_df["Predictions"] - predictions_df["True Values"]
    # Add relative error column:
    predictions_df["Relative Error %"] = predictions_df["Error"] / predictions_df["True Values"] * 100
    # Print in csv file:
    predictions_df.to_csv(os.path.join(dir_mount, "predictions.csv"), index=False)
