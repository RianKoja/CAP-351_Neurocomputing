########################################################################################################################
# Create a final report for the second assignment on Neurocomputing course taken at INPE on 2021.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Standard imports
import os
import re

# PyPI imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from tools import createdocument, graphs
import Regression.tools.manipulations as r_manip
import Classification.tools.manipulations as c_manip

# Start report:
testDoc = createdocument.ReportDocument(title="Second Assignment Report", user_name="Rian Koja")
testDoc.add_heading("CAP-351 - Neurocomputing", level=2)

testDoc.add_heading("Introduction", level=1)
testDoc.add_paragraph(
    "This report presents the results of approaching a regression and a classification problem with Multi-Layer Perceptron networks. The data sets will be presented, along with their separation between training, validation and test subsets. Several topologies are randomly selected for testing, to showcase the accuracies that one may expect to obtain over these data sets and how those should relate amonsgt themselves. The techniques fo momentum and regularization are briefly explained and tested on the best performing topologies found. As requested, all weight optimizations are conducted with the SGD algorithm through this work, despite the ADAM method being one of the most commonly used methods."
)


#######################################################################################################################
# Regression part
#######################################################################################################################

testDoc.add_heading("Regression", level=1)
# Explain what regression is
testDoc.add_heading("What is Regression?", level=2)
testDoc.add_paragraph(
    "A regression problem is a problem consisting of predicting a target variable from some input variables. In other words, trying to predict the value of a function y=f(x) where f is a function of x which cannot be directly computed, and x is an independent variable. The target variable is y, and the input variables are x. It is assumed that several examples of pair (x, y) are available for analysis, and a regression model should be fit for at least most of the available examples."
)

testDoc.add_heading("The Boston Real Estate Dataset", level=2)
# Explain the Boston dataset
testDoc.add_paragraph(
    "The Boston dataset is a collection of 506 data points, including 13 features and 1 target variable. The dataset was collected by the Boston Housing Corporation (BHRC) and was used in the original analysis of the Boston Housing Market. While it is directly available on sci-kit learn's 'dataset' library, it is formatted using the \"DonloadDataset.py\" script on the 'Regression' folder, where further references for this dataset are presented. The data is derived from the 1970 Census of Population and Housing of the United States Census Bureau."
)
testDoc.add_heading("Data Description", level=3)
run = testDoc.add_paragraph("\n")
run.add_run("crim:").bold = True
run.add_run("per capita crime rate by town.\n")
run.add_run("zn: ").bold = True
run.add_run("proportion of residential land zoned for lots over 25,000 sq.ft.\n")
run.add_run("indus: ").bold = True
run.add_run("proportion of non-retail business acres per town.\n")
run.add_run("chas: ").bold = True
run.add_run("Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).\n")
run.add_run("nox: ").bold = True
run.add_run("nitric oxides concentration (parts per 10 million).\n")
run.add_run("rm: ").bold = True
run.add_run("average number of rooms per dwelling.\n")
run.add_run("age: ").bold = True
run.add_run("proportion of owner-occupied units built prior to 1940.\n")
run.add_run("dis: ").bold = True
run.add_run("weighted distances to five Boston employment centres.\n")
run.add_run("rad: ").bold = True
run.add_run("index of accessibility to radial highways.\n")
run.add_run("tax: ").bold = True
run.add_run("full-value property-tax rate per $10,000.\n")
run.add_run("ptratio: ").bold = True
run.add_run("pupil-teacher ratio by town.\n")
run.add_run("black: ").bold = True
run.add_run("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.\n")
run.add_run("lstat: ").bold = True
run.add_run("lower status of the population (percent).\n")
run.add_run("medv: ").bold = True
run.add_run("median value of owner-occupied homes in $1000s. (target variable)\n")

testDoc.add_heading("Surveying Topologies", level=2)

# Import data from the results file in the Regression folder:
df_reg = pd.read_csv(
    os.path.join(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "Regression",
            "mount",
            "Results.csv",
        )
    )
)

testDoc.add_paragraph(
    'The script "SurveyModels.py" iteratively tests random topologies, with some restraints for the number of layers and neurons per layers. It separates the dataset with approximately 70% of the datapoints for training, 20% for validation and 10% for test. The separation is done randomly. The training data is used to adjust the weights of the neural network, the union of test and validation data is used to select the best models based on the R² score, which are saved as pickle files in the "mount" subfolder within the "Regression" directory. The scores for only the training data, only the validation data, the union of train and validation, only test data and all data are saved in the "Results.csv" file located in the mount directory, which stores the results obtained for '
    + str(len(df_reg))
    + " tested topologies."
)


# Plot correlation between the different metrics:

ax, corr = graphs.pandas_correlation(df_reg, "Score Train", "Score Validation")
testDoc.add_fig()

testDoc.add_paragraph(
    "One assumption when separating training and validation data, is that while a neural network may perform poorly on data that was not used to adjust its weights, the most likely reason for this is overtraining, i.e. the network has memorized the examples given to it but nas not encoded any insight into the underlying nature of the phenomena under study. One metric to assess if a network is being trained properly is to compute the correlation between the score obtained in training (which for some datasets and networks the results could be exactly 1)  and the result obtained for the validation data, whose values have no influence over the weights of the neural network. A high positive correlation indicates that overtraining is likely far from occuring in for the tested cases, and the same should apply for diminshing results over the number of attempts to find better topologies. The obtained correlation of "
    + corr
    + "shown in the picture above shows that the network is likely being able to generalize well from the training data into the validation data."
)

ax, corr = graphs.pandas_correlation(df_reg, "Score Train", "Score Dev")
testDoc.add_fig()
testDoc.add_paragraph(
    "Naturally, a higher correlation is expected between the train data set and the development set, which effectively contains 7 data points taken from the traning set for every 9 of its points. The obtained value of "
    + corr
    + " is thus expected and shows that the best models found based on this metric should also generalize well for the test data."
)


ax, corr = graphs.pandas_correlation(df_reg, "Score Dev", "Score Test")
testDoc.add_fig()
testDoc.add_paragraph(
    "Finally, comparing the scores obtained in development against the test data shows a last evidence of how well the models are generalizing. Notably, the scores obtained for the test data should not be used for selection of the mode, so hereinafter are based on the best models found according to the development score."
)

testDoc.add_paragraph("As a final analysis step, it is possible to check the correlation between the development score and the score obtained in the full data set, as shown in the figure below.")


ax, corr = graphs.pandas_correlation(df_reg, "Score Dev", "Score Total")
testDoc.add_fig()
testDoc.add_paragraph(
    "The correlation between the scores obtained in the full dataset and the development dataset is "
    + corr
    + " which shows that the best models found based on this metric should generalize well in the full dataset. This however is a reference for general knowledge and study, and corresponds to a step that would not normally be conducted in the development and deployment of an MLP based product."
)


testDoc.add_heading("Analyzing Convergence and Momentum", level=2)
testDoc.add_paragraph(
    "This model was developed using MLPRegressor class, disabling early stopping and setting a maximum number of iterations to . The maximum number of iterations was set to 2000 and no cases of non-convergence warnings were seen on the console, altough the script that surveys models was left to run unattended in some occasions."
)

# Load a model from the pickle file:
model, layers, layers_str = r_manip.load_pickle()

testDoc.add_paragraph(
    "By default, the models were trained using Nesterov's momentum, and a momentum term of 0.9 which are the default settings of the class MLPRegressor defined in Sci-kit Learn. For comparison, the best model found according to the development score is re-trained from a random initialized state (and the same seed) to that the evolution of the loss curve can be compared. The best topology found at time of creating this report had the internal layers of widths "
    + re.sub(r"(.*),", r"\1,and ", layers_str).replace(",", ", ")
    + ". the input layer has the same size as the number of features and a there is single output neuron."
)
testDoc.add_paragraph(
    "The effect of Nesterov's momentum technique, at least when paired with the 0.9 momentum for gradient descent update, is very minor, a separate figure is shown below to showcase this, however, a minor improvement can be seen by the usage of this more complex technique in terms of convergence obtained over iterations, which may or may not represent a gain in performance depending on the implementation costs of the technique and the activation and updating costs of the network."
)

testDoc.add_paragraph(
    'However, a significant gain is obtained when comparing with not using momentum, which is seen as the "zero momentum" curve, in which not only the speed of convergence is much slower, but also a situation of slow convergence is found much earlier, in which case the end performance of the network after the allowed number of maximum iterations was meaningfully worse.'
)

# Add figure of learning curve obtained when training the loaded model with respect to momentum strategies:
fig, ax = plt.subplots()
ax.plot(model.loss_curve_zeromom, label="No Momentum")
ax.plot(model.loss_curve_09mom, label="0.9 Momentum")
ax.plot(model.clf.loss_curve_, label="Nesterov")
ax.set_title("Learning Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.draw()
testDoc.add_fig()

fig, ax = plt.subplots()
ax.plot([x - y for x, y in zip(model.clf.loss_curve_, model.loss_curve_09mom)], linestyle="dashed", label="No Momentum")
ax.set_title("Learning Curve (Nesterov - 0.9 difference)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Difference")
plt.grid(True)
plt.tight_layout()
plt.draw()
testDoc.add_fig()

testDoc.add_heading("Assessing Regularization", level=2)
testDoc.add_paragraph(
    "Regularization techniques generally consists in adding some inertia to the values of adjusted variables of a system. A classical example is the Levenberg-Marquart method of solving nonlinear systmes of equations, and the same idea is applied with the SGD method. In this case, the changes in weights of the neural network in squared norm are accounted in the loss function to be optimized, but pondered by a regularization factor, in which changing the weights has a cost in itself. This is meant to stabilize the current result of the optimization technique, and avoid oscillations and instabilities."
)

# Add figure of learning curve obtained when training the loaded model with respect to regularization techniques:
fig, ax = plt.subplots()
ax.plot(model.clf.loss_curve_, label="1e-4")
ax.plot(model.loss_reg_zero, label="0")
ax.plot(model.loss_reg_high, label="1e-1")
ax.set_title("Learning Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.draw()
testDoc.add_fig()


testDoc.add_paragraph(
    "In the figure above, it is seen that the default value of 0.0001 for the regularization yields a very comparable result with 0.01, while a meaningful performance loss incurs no regularization is employed, which is basically a sing of poor convergence, which make curve seem trapped in a local minimum of the loss function. This is generally unexpected and happened early in this exercise, while a more common result will be seen in the Classification section."
)

#######################################################################################################################
# Classification part
#######################################################################################################################

testDoc.add_heading("Classification", level=1)
# Explain what regression is
testDoc.add_heading("What is Classification?", level=2)
testDoc.add_paragraph(
    "Classification is the task of assigning or determining a class membership of a given data entry, given its features.  The target variable of a classification task is discrete and referred as a class, label or categorical variable. Members of the same class are expected to have distinctively more similar features between themselves when compared to members of other classes."
)


testDoc.add_heading("The Wine Quality Dataset", level=2)
testDoc.add_paragraph(
    'The wine quality dataset is a collection of 1599 red wine and 4898 white wine samples, each represented by a 13-dimensional feature vector. The usual goal of the wine quality task is to predict the quality of a wine based on the features of the wine. The quality score is a numerical value from 1 to 10, where ten is the highest quality score. The quality score is the target variable of the wine quality task. The white wine is a Portuguese "Vinho Verde", '
)
testDoc.add_paragraph(
    "The dataset is split into two parts, one for red wine and another for (green) white wine. For the current exercise, these datasets were augmented with the class of wine (red: 0 / white: 1) and merged randomly. The new class will be used as the target variable, and thus the goal is predicting is a wine is red or white based on the features of this set and the quality of the wine."
)

testDoc.add_heading("Dataset Description", level=2)
testDoc.add_paragraph(
    "The features of the wine dataset are the 13-dimensional feature vector, which is the 13-dimensional vector representing the wine. The features are the following: "
    + "fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol and quality"
)

# Import data from the results file in the Classification folder
df_class = pd.read_csv(
    os.path.join(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "Classification",
            "mount",
            "Results.csv",
        )
    )
)

testDoc.add_heading("Surveying Topologies", level=2)
testDoc.add_paragraph(
    'The (separate) script "SurveyModels.py" located in the "Classification" folder iteratively tests random topologies, with some restraints for the number of layers and neurons per layers. It separates the dataset with approximately 70% of the datapoints for training, 20% for validation and 10% for test. The separation is done randomly. The training data is used to adjust the weights of the neural network, the union of test and validation data is used to select the best models based on the accuracy obtained by the trained model, which are saved as pickle files in the "mount" subfolder within the "Classification" directory. The scores for only the training data, only the validation data, the union of train and validation, only test data and all data are saved in the "Results.csv" file located in the mount directory, while not apparent on the grpahs, a total of '
    + str(len(df_class))
    + " different topologies were tested. For this exercise, Keras models were used, but the normalization tools from the Sci-kit Learn library were used for convenience."
)


# Plot correlation between the different metrics:

ax, corr = graphs.pandas_correlation(df_class, "Training", "Validation")
testDoc.add_fig()

testDoc.add_paragraph(
    "The scores obtained for the classification task are the accuracies of the models within the dataset, that is, they correspond to the fraction of samples that are correctly classified. Again, the correlation between the training and validation data is presented, but for current results there is little granularity in the results for both the validation and the training data, therefore the data points are located in a grid with intervals related to the fraction that a single data point represents. The obtained correlation of "
    + corr
    + " shown in the picture above reflects this limited representativeness."
)

ax, corr = graphs.pandas_correlation(df_class, "Training", "Development", type="classification")
testDoc.add_fig()
testDoc.add_paragraph(
    "Naturally, a higher correlation is expected between the train data set and the development set, which effectively contains 7 data points taken from the traning set for every 9 of its points. The obtained value of "
    + corr
    + " is thus expected and shows that the best models found based on this metric should once again generalize well for the test data. However, for the limited tests conducted, almost all the results are in a very narrow range, thus allowing to conclude that all the models tested achieved high accuracy."
)


ax, corr = graphs.pandas_correlation(df_class, "Development", "Test", type="classification")
testDoc.add_fig()

testDoc.add_paragraph(
    "Oddly enough, a negative correlation of "
    + str(corr)
    + " was found between the results in validation data and the results in test data. While somewhat unexpected and undesirable, the result is explained by the narrow range of accuracies found for both analysis, since all results were above 99% correct in all cases."
)

ax, corr = graphs.pandas_correlation(df_class, "Development", "Total", type="classification")
testDoc.add_fig()

testDoc.add_paragraph(
    "Despite the previous obsertvation, the correlation between the accuracy found in the development data and the overall data is still quite high at "
    + str(corr)
    + " though the ranges remain small."
)

# Load a model from the pickle file:
model, layers, layers_str = c_manip.load_pickle()

testDoc.add_heading("Assessing momentum strategies", level=2)

testDoc.add_paragraph(
    "Momentum techniques again have been useful in this exercise, but the added complexity of Nesterov's method has show little gain with respect to simply using the standard momentum method with the default value from MLPRegressor's class, altough this network was developed using Keras. This showcases the usefulness of the momentum methods."
)

plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(model.fit_00momentum.history["loss"], label="No momentum")
plt.plot(model.fit_09momentum.history["loss"], label="0.9 Momentum")
plt.plot(model.fit_out.history["loss"], label="Nesterov")
plt.legend()
plt.title("Effect of momentum techniques on loss curve")
plt.tight_layout()
plt.grid(True)
plt.draw()
testDoc.add_fig()

plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss Difference")
plt.plot([x - y for x, y in zip(model.fit_out.history["loss"], model.fit_09momentum.history["loss"])])
plt.title("Difference on loss curve between Nesterov and 0.9 standard momentum", fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.draw()
testDoc.add_fig()


testDoc.add_heading("Assessing Regularization", level=2)

testDoc.add_paragraph(
    "For this classification problem, the effect of regularization was very minor, and barely visible in  the curve shown below. The differences between the loss curves in the base case (i.e. with a regularization of 0.0001) versus a high value of 0.1 and a low value of 0 is shown in a subsequent picture. Different overshooting behaviors are seen while by the end of the iterations the end result seems close to zero. A zoomed picture is shown later which shows that indeed the default regularization term yields the best result, while using no regularization apparently allowed for faster convergence, despite the end result being better when using a high regularization term than none, at least for the applied number of epochs."
)

# Add figure of learning curve obtained when training the loaded model with respect to regularization techniques:
fig, ax = plt.subplots()
ax.plot(model.fit_out.history["loss"], label="1e-4")
ax.plot(model.fit_reg_zero.history["loss"], label="0")
ax.plot(model.fit_reg_high.history["loss"], label="1e-1")
plt.title("Effect of regularization term on loss curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.draw()
testDoc.add_fig()

fig, ax = plt.subplots()
ax.plot([x - y for x, y in zip(model.fit_out.history["loss"], model.fit_reg_zero.history["loss"])], label="1e-4 - 0")
ax.plot([x - y for x, y in zip(model.fit_out.history["loss"], model.fit_reg_high.history["loss"])], label="1e-4 - 1e-1")
plt.title("Differences in loss curves changing regularization term")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.draw()
testDoc.add_fig()

fig, ax = plt.subplots()
ax.plot([x - y for x, y in zip(model.fit_out.history["loss"], model.fit_reg_zero.history["loss"])], label="1e-4 - 0")
ax.plot([x - y for x, y in zip(model.fit_out.history["loss"], model.fit_reg_high.history["loss"])], label="1e-4 - 1e-1")
plt.title("(Zoomed) Differences in loss curves changing regularization term", fontsize=10)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.ylim(-1e-6, 1e-6)
plt.grid(True)
plt.tight_layout()
plt.draw()
testDoc.add_fig()


testDoc.add_heading("Conclusion", level=1)

testDoc.add_paragraph(
    "The score and accuracy results allow noting that several different topologies yielded similarly good results for the two problems taken as examples for this exercise. There haven't been meaningful cases of overfitting found, that is: No very high performance in training cases was coupled with unusually low performance in data not used for training. The configurations tested have shown little effect of the Nesterov's momentum technique, while the standard momentum has shown meaningful gains in convergence speed and end result with respect to not using a gain technique at all. The regularization term has shown a great performance improvement on the example selected for regression, but little to no effect on the classification problem. Possibly, the coupling of regularization with te momentum technique (active in the cases were regularization was varied) is to blame for this, as both methods have a similar role in stabilizing convergence, although theoretically, the regularization would usually slow down convergence while the momentum technique has been shown efficient in accelarating it."
)

testDoc.finish()

print("Finished", __file__)
