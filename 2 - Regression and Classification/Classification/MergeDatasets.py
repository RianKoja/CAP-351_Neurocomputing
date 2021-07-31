########################################################################################################################
# Merge the datasets of red and white (green) wine into a single dataset for usage in classification
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports
import os

# PyPI imports
import pandas as pd
import numpy as np

# Set the directory where data is stored
dir_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset")

# Create dataset for red wines:
red_df = pd.read_csv(os.path.join(dir_data, "winequality-red.csv"), sep=";")
red_df["target"] = 0

# Create dataset for white wines:
white_df = pd.read_csv(os.path.join(dir_data, "winequality-white.csv"), sep=";")
white_df["target"] = 1

# Merge datasets:
df = red_df.append(white_df)

# Use fixed random seed to have well defined behavior
np.random.seed(2626354877)

# Reshuffle the dataset:
df = df.sample(frac=1).reset_index(drop=True)

# Save the dataset to a xlsx file:
df.to_excel(os.path.join(dir_data, "winequality-merged.xlsx"), index=False)
