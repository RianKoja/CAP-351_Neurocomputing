########################################################################################################################
# Separate data into training, validation and test sets
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports:
import glob
import os
import pickle
import re
import sys

# PyPI imports:
import numpy as np


# Split data into training, validation and test sets
def split_data(model, x_total, y_total, proportion_train=0.7, proportion_validation=0.2):

    # Ensure seed is fixed, so separation is always done the same way:
    np.random.seed(2626354877)

    train_length = round(proportion_train * model.n_samples)
    validation_length = round(proportion_validation * model.n_samples)

    index_train = np.random.choice(range(0, len(x_total)), train_length, replace=False)
    index_remaining = [n for n in range(model.n_samples) if n not in index_train]
    index_valid = np.random.choice(range(0, len(index_remaining)), validation_length, replace=False)
    index_tests = [n for n in index_remaining if n not in index_valid]
    index_dev = [*index_train, *index_valid]

    x_train = x_total[index_train, :]
    y_train = y_total[index_train]
    x_valid = x_total[index_valid, :]
    y_valid = y_total[index_valid]
    x_dev = x_total[index_dev, :]
    y_dev = y_total[index_dev]
    x_tests = x_total[index_tests, :]
    y_tests = y_total[index_tests]

    return x_train, y_train, x_valid, y_valid, x_dev, y_dev, x_tests, y_tests


# Use a function to load the pickle file, which can be imported elsewhere:
def load_pickle(pickle_file=None):
    # Define folder used for handling files:
    dir_mount = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "mount")

    # Select model file:
    if pickle_file is None:
        # Find latest model found:
        list_of_files = glob.glob(os.path.join(dir_mount, "*).pckl"))
        latest_file = max(list_of_files, key=os.path.getctime)
        model_picklefile = latest_file
    else:
        model_picklefile = pickle_file

    # Load the model:
    with open(model_picklefile, "rb") as fid:
        print("Loading ", model_picklefile)
        try:
            model = pickle.load(fid)
        except ModuleNotFoundError:
            fid.seek(0)
            sys.path.append("Regression")
            model = pickle.load(fid)
        layers_str = re.search(r"\(([^)]+)", model_picklefile).group(1)
        layers = tuple([int(x) for x in layers_str.split(",")])

    return model, layers, layers_str
