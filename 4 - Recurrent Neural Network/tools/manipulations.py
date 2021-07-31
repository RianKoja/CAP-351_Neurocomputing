# Standard imports:
import os

# PyPI imports:
import numpy as np
import pandas as pd


def get_tickers():
    raw_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "RawData")
    tickers_file = os.path.join(raw_data_folder, "Tickers.xlsx")
    # Create a list of tickers from the Tickers.xlsx file:
    tickers_list = []
    file_path = os.path.join(raw_data_folder, tickers_file)
    for sheet in pd.ExcelFile(tickers_file).sheet_names:
        # Read data from excel into a dataframe without header:
        temp_data = pd.read_excel(file_path, sheet_name=sheet, header=None)
        # Add the tickers to the list:
        temp_list = temp_data[0].to_list()
        tickers_list += temp_list

    return tickers_list


# Create the number of hidden layers and neurons per layer randomly, with limitations:
def make_random_layers(min_input=7):
    this_length = np.random.randint(0, 10)  # Up to ten hidden layers
    num_list = []
    for _ in range(this_length + 1):
        num_list.append(np.random.randint(20, 800))  # Range of neurons per layer

    if num_list[0] < min_input:
        num_list[0] = min_input

    return tuple(num_list)


# convert integer to representation in base n:
def int2base(x, n):
    if x < n:
        return [x]
    else:
        return int2base(x // n, n) + [x % n]


def make_sequential_layers(enumerator, base, size_input=7, size_output=1):
    # Start a tuple with input size:
    layers = [size_input]
    # convert integer to representation in base n:
    layers_str = int2base(enumerator, base)
    # Add hidden layers:
    for c in layers_str:
        layers.append(1 + int(c))
    # Add last layer
    layers.append(size_output)

    return tuple(layers)


def str2list(eg):
    return [int(k) for k in eg.replace("(", "").replace(")", "").replace(" ", "").split(",") if k != ""]


# Compute number of parameters in a model:
def complexity_count(layers):
    # One bias per neuron, except for hidden layer:
    param_count = sum(layers[1:])
    # One weight per connection between each layer:
    for ii in range(0, len(layers) - 1):
        # One weight per connection
        param_count += layers[ii] * layers[ii + 1]

    return param_count


def disp_complexity(n_features, layers):
    print(f"f({n_features}, {layers}) = {complexity_count(layers)}")


if __name__ == "__main__":
    n_features = 7
    topologies = list(set(make_sequential_layers(n, 25, size_input=n_features) for n in range(0, 25 ** 5)))
    print("Topologies built!")
    complexities = [complexity_count(topology) for topology in topologies]
    print("Complexities computed!")
    df = pd.DataFrame({"Complexity": complexities, "Layers": topologies})
    df.sort_values("Complexity", ascending=True, inplace=True)

    csv_filename = os.path.join(os.path.dirname(__file__), "..", "Models", "Layers_by_complexity.csv")
    df.to_csv(csv_filename, index=False)

    print("Finished", os.path.basename(__file__))
