########################################################################################################################
# Create some useful plots.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################


# PyPI imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def pandas_correlation(df, x_label, y_label, type="regression"):
    ax = df.plot.scatter(x=x_label, y=y_label, label="Scores")
    slope, intercept, rvalue, pvalue, stderr = linregress(df[x_label].to_list(), df[y_label].to_list())
    df["Regression"] = intercept + slope * df[x_label]
    df.plot(x=x_label, y="Regression", ax=ax, color="Red", label="Regression")
    plt.legend(loc="upper left")
    if type == "regression":
        plt.title("Correlation between R² score for training and validation data\n Correlation: %.2f" % rvalue)
        plt.xlabel("R² score for " + x_label)
        plt.ylabel("R² score for " + y_label)
    else:
        plt.title("Correlation between accuracy for training and validation data\n Correlation: %.2f" % rvalue)
        plt.xlabel("Accuracy for " + x_label + " data")
        plt.ylabel("Accuracy for " + y_label + " data")
    plt.grid(True)
    plt.tight_layout()
    plt.draw()

    return ax, " %.2f" % rvalue
