import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from copy import copy, deepcopy
from sklearn.model_selection import ParameterGrid
import random
from sklearn.metrics.pairwise import euclidean_distances
import os
import pickle

from sklearn.preprocessing import StandardScaler


def standardise_data(x_train, x_val, x_test):
    '''
    Standardise train data to have zero mean and standard deviation of 1.
    Validation and test data standardised by the same parameters as train data.

    Inputs:
    - x_train - train data (np.array or pd.DataFrame()
    - x_val - val data
    - x_test - test data

    Outputs:
    - transformed datasets in the same order
    '''
    scaler = StandardScaler()

    scaler.fit(x_train)

    train_df = pd.DataFrame(scaler.transform(x_train))
    val_df = pd.DataFrame(scaler.transform(x_val))
    test_df = pd.DataFrame(scaler.transform(x_test))

    return (train_df, val_df, test_df)


def fit_and_predict(model, x_train, x_test, y_train):
    '''
    For a given model, fit it using the train data and form predictions based on test data and calculate fitted value based on train data.

    Inputs:
    - model: Model object. Must have.fit() method
    - x_train: train features
    - x_test: test features
    - y_train: train targets

    Outputs:
    - y_predict_train: fitted values
    - y_predict_test: predictions on the test set
    - model: fitted model object
    '''
    model.fit(x_train, y_train)
    y_predict_train = model.predict(x_train)
    y_predict_test = model.predict(x_test)

    return (y_predict_train, y_predict_test, model)


def r2_score_loss(y_true, y_pred):
    '''
    For a given prediction and true labels, calculate negative of R-squared.
    '''
    r = -r2_score(y_true, y_pred)
    return (r)


def model_tuning(model, par_grid, x_train, y_train, x_val, y_val, metric=r2_score):
    '''Fit and predict using a given models with a given parameter grid
    on train (fit & predict) and validation data (predict only).

    Inputs:
    - parameter grid: dictionary of parameters with attribute names as keys and values as items. For example,
    for a random forest model:

    rf_grid = {"n_estimators": [20, 40, 60],
          "max_depth": [5, 10, 15, 20]}

    - x_train: training set of features,
    - y_train: training set of labels
    - x_val: validation set of features
    - y_val: validation set of labels

    - metric: note that this needs to be in the form of loss, i.e. lower values are better.

    Returns:
    - dictionary of train and validation performance for different parameter values
    - best model as measured by validation set performance (object)
    - dictionary of predictions on train and val set for the top model'''

    # reshape response if needed
    if y_train.ndim > 1:
        y_train = np.array(y_train).reshape(-1)
    else:
        pass

    results = dict()  # dictionary to store results

    i = 0

    best_metric = +np.inf
    best_model = np.nan
    best_model_predictions = dict()

    # Loop through models defined in the grid
    for model_spec in list(ParameterGrid(par_grid)):
        print(model_spec)
        print("model iteration", i)
        i += 1

        model_instance = deepcopy(model)
        model_name = ""

        # set up attributes
        for attribute, value in model_spec.items():
            model_name = model_name + str(attribute) + ": " + str(value) + " "
            setattr(model_instance, attribute, value)  # set attribute values as specified in the grid

        print(model_instance)  # print to check

        # fit & predict
        y_predict_train, y_predict_val, model = fit_and_predict(model_instance, x_train, x_val, y_train)

        # metric results
        results[model_name] = dict()

        val_metric = metric(y_val, y_predict_val)

        results[model_name]["train_metric"] = np.round(metric(y_train, y_predict_train), 3)
        results[model_name]["val_metric"] = np.round(val_metric, 3)

        # keep track of the best model
        if val_metric < best_metric:
            print("New best metric:", val_metric)
            best_metric = val_metric
            best_model = model_instance
            best_model_predictions["y_predict_train"] = y_predict_train
            best_model_predictions["y_predict_val"] = y_predict_val

    return (results, best_model, best_model_predictions)


def model_tuning_cv(model, param_grid, x_train, y_train, x_test, score=None, cv=None):
    '''Fit and predict using cross-validation on train set.

    Inputs:
    - parameter grid: dictionary of parameters with attribute names as keys and values as items. For example,
    for a random forest model:

    rf_grid = {"n_estimators": [20, 40, 60],
          "max_depth": [5, 10, 15, 20]}

    - x_train: training set of features,
    - y_train: training set of labels
    - x_test: test set of features

    - score: one of Python's metric methods
    - cv: CV object

    Returns:
    - dictionary of CV results
    - best model as measured by CV
    - dictionary of predictions on test set'''

    # reshape response if needed
    if y_train.ndim > 1:
        y_train = np.array(y_train).reshape(-1)
    else:
        pass

    i = 0

    parameters = ParameterGrid(param_grid).__dict__["param_grid"][0]
    clf = GridSearchCV(model, parameters, cv=cv, scoring=score)
    clf.fit(x_train, y_train)

    best_model = clf.best_estimator_
    cv_results = clf.cv_results_

    best_model_predictions = dict()

    model_train = best_model.fit(x_train, y_train)
    best_model_predictions["y_test_predict"] = best_model.predict(x_test)

    return (cv_results, best_model, best_model_predictions)




def calculate_reconstr_loss_pca_2(pca, x_original, n_comp, trim=False, rescale=False, mean=None, std=None):
    '''
    Calulcate reconstruction loss for PCA with a given number of components.

    Inputs:
    - x_original: dataset on which to calculate the reconstruction loss (usually validation data)
    - n_comp: number of principal components to use
    - pca: estimated pca object (i.e. after calling the "fit" method)
    - trim: (True/False) should PCAreconstruction be trimmed at zero
    - rescale: Ture if loss should be calculated in the original scale
    - mean: supply if rescale = True
    - std: supply if rescale = True

    Returns:
    - loss
    - reconstructed dataset
    - transformed dataset'''

    x_projected = x_original @ pca.components_[:n_comp, :].T
    x_original_space = x_projected @ pca.components_[:n_comp, :]

    # Trimming done in the rescaled (original) scale, hence trim = True requires rescaled = True
    #     if (trim == True) & (rescale == False):
    #         rescale = True
    #         print( "Warning! Trimmed set to true, hence setting rescale to True." )
    #     else:
    #         pass

    if rescale == True:
        x_original = rescale_back(x_original, mean, std)
        x_original_space = rescale_back(x_original_space, mean, std)
    else:
        pass

    if trim == True:
        x_original_space[x_original_space < 0] = 0
    else:
        pass

    loss = ((x_original - x_original_space) ** 2).sum().sum() / x_original.size

    return (x_projected, x_original_space, loss)

def compare_distances_pca(x, x_transformed, sample_size = 100):
    '''Calculate distance ratio between original and projected datapoints.

    Return projected_distances/original_distances, for the supplied original dataset (x) and transformed datase (x_transformed),
    based on a "sample_size" randomly drawn sample points.

    Input:
    - x: original dataset (numpy array or dataframe)
    - x_transformed : transformed/projected dataset (numpy array or dataframe)
    (the order in which they are supplied affects only their place in the final ratios).
    - sample_size: number of points to consider when calculating distance ratios (integer).

    Return:
    - mean and standard deviation of ratios between projected and original distances
    (i.e. ratio = proejcted distance / original distance)

    '''

    sample = random.sample(np.arange(0, x.shape[0], 1).tolist(), sample_size)

    # In case data were supplied as dataframes, convert them to numpy arrays
    try:
        x = x.to_numpy()
        x_transformed = x_transformed.to_numpy()
    except:
        pass

    x_to_compare = x_transformed[sample, :]

    # Calculate distances on the original data
    # Drop zero distances (i.e. those corresponding to distance between a point and itself)
    distances_original = euclidean_distances(x[sample, :], squared=True).ravel()
    nonzero_original = distances_original  != 0
    distances_original  = distances_original[nonzero_original]

    # Calculate distances on the projected data
    distances_projected = euclidean_distances(x_to_compare, squared=True ).ravel()
    nonzero_projected = distances_projected  != 0
    distances_projected  = distances_projected[nonzero_projected]

    # Scaling to account for different dimensionality
    distances_projected = np.sqrt( x.shape[1]/x.shape[0] )*distances_projected

    # calculate distance ratios
    ratios = distances_projected / distances_original

    return( ratios.mean(), ratios.std())



def rescale_back(observation, mean, std):
    '''
    Reverse standardisation using the given parameters

    Inputs:
    - observation: numpy array (zero mean and std of 1, can be 2D if "mean" and "std" have compatible dimensions)
    - mean
    - std

    Returns:
    - np array: observation*std + mean
    '''
    return( observation*std + mean )
#     return( observation + mean )


def calculate_std_abs_loadings(loadings, start_component, stop_component):
    '''
    Calulcate total absolute loadings of components starting with
    start_component and ending with stop_component -1.
    This is as opposed to calculating loadings of components 1: (stop_component - 1).

    The loadings are standardise (divided by maximum value) to be below 1'''
    loadings_abs_pca_top = np.abs(loadings)[start_component:stop_component, :].T.sum(axis=1)
    loadings_abs_pca_top = (loadings_abs_pca_top - loadings_abs_pca_top.min()) / loadings_abs_pca_top.max()

    return (loadings_abs_pca_top)



def numpy_rolling_mean(np_array, window, min_periods = None, index = None):
    '''
    Return a rolling average of a given numpy array. The average is in the middle of the window,
    i.e. for window of 20, average is calculated from 10 values to the left and 10 to the right of a data point

    Inputs:
    - np_array
    - window: integer - window used for calculating the average
    - min_periods: min size of the window for which to still calculate the average (affects corner points)
    - index: index of the input series (if any)

    Returns:
    - Series of rolling averages
    '''
    df_series = pd.DataFrame(np_array)
    df_series.set_index(index, inplace = True)
    return( df_series.rolling(window, min_periods, center = True).mean() )


def plot_coefs_vs_wavelength(coefs, window, min_periods=None, title=None, xticks=None,
                             color=None, color_ma=None, y_lim=[None, None]):
    '''
    Plot coefficients and rolling mean of coefficients vs wavelengths.

    Inputs:
    - coefs: coefficients to plot.
    - window: window to calculate rolling means (for window of 20, average is calculated from 10 observations to the left and 10 to the right).
    - min_periods: minimum window size for which to still calculate average (concerns corner points)
    - title: title of the plot
    - xticks: custom xticks if needed (these should be wavelengths)
    - color of the coefficients
    - color of the rolling mean of coefficients
    - y_lim: limits on y axis as in matplotlib, e.g. [0.0, 1.0]

    Returns:
    - empty
    '''
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(xticks, coefs, linewidth=0.1, color=color)
    ax.tick_params(labelsize=18)
    ax.plot(numpy_rolling_mean(coefs, window, min_periods, xticks),
            linewidth=0.8,
            color=color_ma);
    ax.set_ylim(y_lim)

    if title: ax.set_title(title, fontsize=20);
    return ()


def calculate_reconstr_loss_pca_tau(pca, x_original, n_comp, rescale=False, mean=None, std=None):
    '''
    Calulcate reconstruction loss for PCA with a given number of components.

    Inputs:
    - x_original: dataset on which to calculate the reconstruction loss (usually validation data)
    - n_comp: number of principal components to use
    - pca: estimated pca object (i.e. after calling the "fit" method)

    Returns:
    - transformed dataset
    - reconstructed dataset
    - transformed dataset
    - loss
    '''

    x_projected = x_original @ pca.components_[:n_comp, :].T
    x_original_space = x_projected @ pca.components_[:n_comp, :]

    if rescale == True:
        x_original = rescale_back(x_original, mean, std)
        x_original_space = rescale_back(x_original_space, mean, std)
    else:
        pass

    loss = ((np.exp(-x_original) - np.exp(-x_original_space)) ** 2).sum().sum() / x_original.size

    x_projected = np.exp(-x_projected)

    x_original_space = np.exp(-x_original_space)

    return (x_projected, x_original_space, loss)


def calculate_reconstr_loss_spca(spca_dict, x_original, n_comp, mean, std):
    '''
    Given a dictionary of spca object (recovered from the pickle file), original data and number of
    principal components to use, calculate MSE reconstruction loss.

    Inputs:
    - spca_dict: loaded dictionary of spca model
    - x_original: original dataset
    - n_comp: number of components to use
    - mean: mean used for standardisation of data
    - std: std used for standardisation of data
    '''
    x_projected = x_original @ spca_dict["spca_object"].components_[:n_comp, :].T
    # Transform projected data back to the original space
    x_original_space = x_projected @ spca_dict["spca_object"].components_[:n_comp, :]  # project back

    x_original = rescale_back(x_original, mean, std)
    x_original_space = rescale_back(x_original_space, mean, std)

    loss = ((x_original - x_original_space) ** 2).sum().sum() / x_original.size

    return (loss)


def dump_object(object_to_save, location, filename):
    '''Saves a given object under a given name in a given location / direcotry (if such directory
     does not exist the function creates it.

     Inputs:
     - object_to_save: object to save.
     - location: directory/folder where to save the object
     - filename: how to name the saved file

     Returns:
     - "": empty. The function performs action, returns nothing.
     '''
    if not os.path.isdir(location):
        try:
            os.mkdir(location)
        except:
            print("Failed")

    file_path = location + "/" + filename

    with open(file_path, 'wb') as dump_location:
        pickle.dump(object_to_save, dump_location)

    return ()