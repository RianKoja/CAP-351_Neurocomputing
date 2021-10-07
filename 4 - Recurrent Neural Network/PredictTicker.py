########################################################################################################################
# Train model for predicting a specified stock ticker's low price 20 financial days in the future days based on the
# sequence of days until given date. Uses the model in the ModelSpecs class. Evaluates different topologies searching
# for the smallest viable one.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports:
import copy
import os
import pickle
import time

# PyPI imports:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports:
from tools import classes, PlotData


print("Starting", os.path.basename(__file__))

data_folder = os.path.join(os.path.dirname(__file__), "ProcessedData")

ticker = "EMBR3"  # "AZUL4"  #  "SUZB3"  #  "MGLU3"  #  "ENGI4"

data_file = os.path.join(data_folder, ticker + ".csv")

mdl_empty = classes.ModelSpecs(data_file)

x, y, dt = mdl_empty.get_x_y(data_file)

x_train, y_train, dt_train, x_test_ini, y_test_ini, dt_test_ini, x_test_end, y_test_end, dt_test_end = PlotData.split_sets(x, y, dt)


# A list of possible topologies, ordered by complexity (number of parameters if MLP) is created with tools.manipulations.py:
layers_file = os.path.join(os.path.dirname(__file__), "Models", "Layers_by_complexity.csv")
layers_df = pd.read_csv(layers_file)

plotting = False
mdl = None
# Create Log:
csv_name = os.path.join(os.path.dirname(__file__), "Results", "Results_" + ticker + ".csv")
if os.path.isfile(csv_name):
    df_scores = pd.read_csv(csv_name, index_col=None)
    mse_min = min(df_scores["MSE Train and Test"])
else:
    df_scores = pd.DataFrame([], columns=["Layers", "Complexity", "MSE Train", "MSE Future", "MSE Past", "MSE Train and Test"], index=None)
    mse_min = np.inf


for index, row in layers_df.iterrows():
    complexity = row["Complexity"]
    layers_str = row["Layers"]
    if not (df_scores["Layers"] == layers_str).any():
        layers = tuple(map(int, layers_str[1:-1].split(", ")))
        hidden_layers = layers[1:-1]
        if 1 in hidden_layers or 2 in hidden_layers or 3 in hidden_layers or 4 in hidden_layers or 5 in hidden_layers or 6 in hidden_layers or 8 in hidden_layers or 9 in hidden_layers or 10 in hidden_layers:
           continue
        print(f"Working on complexity = {complexity}, layers = {layers} hidden_layers = {hidden_layers}")

        del mdl
        mdl = copy.deepcopy(mdl_empty)

        mdl.train(x_train, y_train, x_test_end, y_test_end, layers)
        mdl.complexity = complexity

        if plotting:
            plt.figure(figsize=(10, 6))
            plt.plot(mdl.fit_out.epoch, mdl.fit_out.history["loss"], label="Training")
            plt.plot(mdl.fit_out.epoch, mdl.fit_out.history["val_loss"], label="Validation")
            plt.grid("minor", "both")
            plt.title(f'Loss during training for topology: {layers_str}, Complexity: {complexity}, training loss: {mdl.fit_out.history["loss"][-1]}')
            plt.legend()
            plt.tight_layout()
            plt.draw()

        # print(f'loss vs score: loss={mdl.fit_out.history["loss"][-1]}, score = {mdl.score(x_train, y_train)}')

        mse_train_and_test = mdl.score(np.concatenate([x_train, x_test_end]), np.concatenate([y_train, y_test_end]))

        # Save model with pickle if best accuracy found so far:
        if mse_min > mse_train_and_test:
            mse_min = mse_train_and_test
            mdl.predict_output_trn = mdl.predict(x_train)
            mdl.predict_output_ini = mdl.predict(x_test_ini)
            mdl.predict_output_end = mdl.predict(x_test_end)
            model_filename = os.path.join(os.path.dirname(__file__), "Models", ticker + layers_str + ".pckl")
            with open(model_filename, "wb") as fid:
                pickle.dump(mdl, fid)

        df_scores.loc[len(df_scores)] = [layers, complexity, mdl.score(x_train, y_train), mdl.score(x_test_end, y_test_end), mdl.score(x_test_ini, y_test_ini), mse_train_and_test]

        df_scores.to_csv(csv_name, index=False)

        if plotting:
            # Evaluate model:
            predict_output_trn = mdl.predict(x_train)
            predict_output_ini = mdl.predict(x_test_ini)
            predict_output_end = mdl.predict(x_test_end)

            df_test_ini = copy.deepcopy(dt_test_ini)
            df_test_ini.insert(1, "y", y_test_ini)
            df_test_ini.insert(2, "y_pred", predict_output_ini)

            df_train = copy.deepcopy(dt_train)
            df_train.insert(1, "y", y_train)
            df_train.insert(2, "y_pred", predict_output_trn)

            df_test_end = copy.deepcopy(dt_test_end)
            df_test_end.insert(1, "y", y_test_end)
            df_test_end.insert(2, "y_pred", predict_output_end)

            # Plot model results against test data:
            ax = df_test_end.plot(x="Date", y=["y", "y_pred"], label=["y_test_end", "y_pred_end"], figsize=(10, 6))
            df_train.plot(x="Date", y=["y", "y_pred"], label=["y_train", "y_pred_train"], ax=ax)
            df_test_ini.plot(x="Date", y=["y", "y_pred"], label=["y_test_ini", "y_pred_ini"], ax=ax)
            plt.title(f'Predictions vs reference for topology: {layers_str}, Complexity: {complexity}, training loss: {mdl.fit_out.history["loss"][-1]}')
            plt.grid("minor", "both")
            plt.tight_layout()
            plt.draw()

            mdl.plot_model()
            os.system(f'start "" "{mdl.layers}.png"')

    # if complexity > 31:
    #   break  # Suffices for debugging

if plotting:
    print("Before plt.show, close graphs to finish program.")
    plt.show()

print("Finished", os.path.basename(__file__))
