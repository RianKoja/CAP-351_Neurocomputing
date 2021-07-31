########################################################################################################################
# Define an auxiliary function `str2list` and a main function `complexity_count`
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################


def str2list(eg):
    #    return list(map(int, eg.replace('(', '').replace(')', '').split(',')))
    return [int(k) for k in eg.replace("(", "").replace(")", "").replace(" ", "").split(",") if k != ""]


def complexity_count(data_frame, model):
    param_count_list = []
    for nn in range(0, len(data_frame)):
        # Compute number of parameters per model:
        layer = str2list(data_frame["Layers"][nn])
        # One weight per connection between input and first layer, then one bias per neuron plus one weight per
        # connection between last layer and output:
        param_count = model.n_features * layer[0] + sum(layer) + layer[-1]
        for ii in range(0, len(layer) - 1):
            # One weight per connection
            param_count += layer[ii] * layer[ii + 1]
        param_count_list.append(param_count)

    # Add Model Parameters column
    data_frame["Parameter Count"] = param_count_list
    return data_frame
