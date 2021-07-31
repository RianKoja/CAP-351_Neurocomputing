
# Standard imports
from genericpath import exists
import itertools
import os
import pickle

# PyPI imports:
import matplotlib.pyplot as plt
import minisom
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Local imports
from tools import fma_utils, classes, loader

print(f'Starting {os.path.basename(__file__)}')

data_dir = os.path.join(os.path.dirname(__file__), "dataset")
mount_dir = os.path.join(os.path.dirname(__file__), "mount")
for dirs in [data_dir, mount_dir]:
    if not os.path.exists(dirs):
        os.mkdir(dirs)

features, echonest, genres_full, genres_echo = loader.load_datasets()


# Set multiple parameters for SOM:
x_y_set = (10, 20, 40, 80)
sigma_set = (5, 3, 1)
learning_rate_set = (0.8, 0.5, 0.2)

# Prepare scalers:
ss_features = StandardScaler()
ss_echonest = StandardScaler()
X_features = ss_features.fit_transform(X=features.to_numpy())
X_echonest = ss_echonest.fit_transform(X=echonest.to_numpy())

for df_np, df_name, input_len, ss, genres in [(X_features, 'features', len(features.columns), ss_features, genres_full), (X_echonest, 'echonest', len(echonest.columns), ss_echonest, genres_echo)]:
    print(df_name)
    for x_y, sigma, learning_rate in itertools.product(x_y_set, sigma_set, learning_rate_set):
        x = y = x_y
        # Create a SOM for the features dataframe:
        mdl = classes.SomClass(x, y, input_len, sigma, learning_rate, df_name)
        pckl_name = os.path.join(mount_dir, mdl.name + '.pckl')

        if os.path.exists(pckl_name):
            continue
        mdl.scaler = ss
        mdl.som.train_random(data=df_np, num_iteration=20000)
        mdl.topographic_error = mdl.som.topographic_error(df_np)

        # Print a table with the most common top genre in each neuron
        # Initialize pandas dataframe with empty strings:
        top_genres = pd.DataFrame(np.zeros((mdl.x, mdl.y)), dtype=str)
        labels = mdl.som.labels_map(df_np, genres)
        
        for j in range(mdl.x):
            for k in range(mdl.y):
                aux = labels[(j, k)]
                if aux:
                    aux.pop(np.nan, None)
                    if aux:
                        daux = max(aux, key=aux.get)
                        top_genres.iloc[j, k] = daux
                    else:  # If empty
                        top_genres.iloc[j, k] = 'None'
                else:  # If empty
                    top_genres.iloc[j, k] = 'None'
                
        mdl.genres_df = top_genres

        # Use pickle to save the trained som to a file identified by the parameters used to train it:
        with open(pckl_name, 'wb') as f:
            pickle.dump(mdl, f)



print(f'Finished {os.path.basename(__file__)} still need to close graphs before re-running')
