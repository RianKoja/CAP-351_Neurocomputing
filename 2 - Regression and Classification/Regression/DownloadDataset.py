########################################################################################################################
# Downloads a dataset from sklearn library, and saves it to a local directory as an excel spreasheet.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports:
import os

# PyPI imports:
import pandas as pd
from sklearn.datasets import load_boston

# Load data from sklearn library:
data_set_raw = load_boston()

# Create a pandas dataframe with the fields in data_set_raw.feature_names and values in data_set_raw.data
df = pd.DataFrame(data_set_raw.data, columns=data_set_raw.feature_names)

# Add a last column coresponding to the target value
df["target"] = data_set_raw.target

# save dataset to a xlsx file:
file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mount", "boston_dataset.xlsx")
df.to_excel(file_name, sheet_name="sheet1", index=False)

# Dataset based on census of population and housing
# ref: https://www.census.gov/library/publications/1972/dec/phc-1.html
# Extracted from
#    Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
#    Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
