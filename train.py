
"""
Prepare and train and save a model.
"""



import pandas as pd





"""
LAUNCH THIS ON SACAPUS MASSIVELY, ALONG WITH THE CREATION OF THE OTHER CELL LINES, WITH A MAXIMUM OF CORES
"""


def prepare_model_with_params(params):
    """
    A wrapper function that will prepare an atyPeak model given the current parameters.
    Non-pure, since it depends on the rest of the code.

    Useful for grid search.
    """

    # Put everything between ### MODEL ### and #-- Training--#


def train_model(model,params):
    """
    A wrapper function that will train a given model given the current parameters.
    Non-pure, since it depends on the rest of the code.

    Useful for grid search.
    """

    # All the  #-- Training--# section

    return trained_model


# Grid search part
# Put it in a commented block only

# Read the yaml parameters to get a default parameters

parameters_to_try = []
# Copy the original parameters and replace only the part we want
parameters_custom = copy(params)
parameters_custom[key] = new_value
parameters_to_try += [parameters_custom]

result_grid = pd.DataFrame()

# now do the search
for parameters_custom in parameters_to_try:
    model = prepare_model_with_params(parameters_custom)
    trained_model = train_model(model, parameters_custom)
    q_score, _,_,_ = q_score(model, train_generator)

    # Add resulting q-score to parameters
    parameters_to_try.update({'Q_score':q_score})

    parameters_to_try = parameters

    # And record the q_score
    result_grid = result_grid.append(pd.Series(parameters_to_try), ignore_index = True)
