import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix
import regression_models as rm
from itertools import product
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from itertools import product
from collections import OrderedDict
from time import localtime, time

def CVpartition(X, y, Type = 'Re_KFold', K = 5, Nr = 10, random_state = 0, group = None):
    """
    Partitions data for cross-validation and bootstrapping.
    Returns a generator with the split data.

    Parameters
    ----------
    model_name : str
        Which model to use
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    Type : str, optional, default = 'Re_KFold'
        Which cross validation method to use.
    K : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    random_state : int, optional, default = 0
        Seed used for the random number generator.
    group : list, optional, default = None
        Group indices for grouped CV methods.
    """
    Type = Type.casefold() # To avoid issues with uppercase/lowercase
    if Type == 'mc':
        CV = ShuffleSplit(n_splits = Nr, test_size = 1/K, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'single':
        X, X_test, y, y_test = train_test_split(X, y, test_size = 1/K, random_state = random_state)
        yield (X, y, X_test, y_test)
    elif Type == 'kfold':
        CV = KFold(n_splits = int(K))
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'stratifiedkfold':
        CV = StratifiedKFold(n_splits = int(K), shuffle = True, random_state = 123)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 're_kfold':
        CV = RepeatedKFold(n_splits = int(K), n_repeats = Nr, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'timeseries':
        TS = TimeSeriesSplit(n_splits = int(K))
        for train_index, val_index in TS.split(X):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'single_group':
        label = np.unique(group)
        num = int(len(label)/K)
        final_list = np.squeeze(group == label[0])
        for i in range(1, num):
            final_list = final_list | np.squeeze(group == label[i])
        yield(X[~final_list], y[~final_list], X[final_list], y[final_list])
    elif Type == 'group':
        label = np.unique(group)
        for i in range(len(label)):
            yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'group_no_extrapolation':
        label = np.unique(group)
        for i in range(len(label)):
            if min(label) < label[i] and label[i] < max(label):
                yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'groupkfold':
        gkf = GroupKFold(n_splits = int(K))
        for train_index, val_index in gkf.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'groupshufflesplit':
        gss = GroupShuffleSplit(n_splits = int(Nr), test_size = 1 / K, random_state = random_state)
        for train_index, val_index in gss.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'no_cv':
        yield (X, y, X, y)
    elif Type == 'single_ordered':
        num = X.shape[0]
        yield (X[:num-round(X.shape[0]*1/K):], y[:num-round(X.shape[0]*1/K):], X[num-round(X.shape[0]*1/K):], y[num-round(X.shape[0]*1/K):])
    else:
        raise ValueError(f'{Type} is not a valid CV type.')

def CV_mse(model_name, X, y, X_test, y_test, X_unscaled = None, y_unscaled = None, cv_type = 'Re_KFold', K_fold = 5, Nr = 10, eps = 1e-4, group = None, **kwargs):
    """
    Determines the best hyperparameters using MSE based on information criteria.
    Also returns MSE and yhat data for the chosen model.

    Parameters
    ----------
    model_name : str
        Which model to use
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    X_unscaled, y_unscaled : Numpy array with shape N x m, N x 1
        The unscaled values of X and y, used during CV to avoid validation set leakage
        Used when model_name != 'LCEN', as it already uses unscaled inputs
    cv_type : str, optional, default = None
        Which cross validation method to use.
    K_fold : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    eps : float, optional, default = 1e-4
        Tolerance. TODO: expand on this.
    **kwargs : dict, optional
        Non-default hyperparameters for model fitting.
    """
    # Setting up some general kwargs
    if 'robust_priority' not in kwargs: # This should not be the case unless the user called this function manually, which is not recommended
        kwargs['robust_priority'] = False
    if 'l1_ratio' not in kwargs:
        kwargs['l1_ratio'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    if 'alpha' not in kwargs: # Unusual scenario, since SPA passes a default kwargs['alpha'] == 20
        kwargs['alpha'] = np.concatenate(([0], np.logspace(-4.3, 0, 20)))
    elif isinstance(kwargs['alpha'], int): # User passed an integer instead of a list of values
        kwargs['alpha'] = np.concatenate( ([0], np.logspace(-4.3, 0, kwargs['alpha'])) )
    if 'use_cross_entropy' not in kwargs:
        kwargs['use_cross_entropy'] = False
    if 'verbosity_level' not in kwargs:
        kwargs['verbosity_level'] = 2

    if model_name == 'EN':
        hyperparam_prod = list(product(kwargs['l1_ratio'], kwargs['alpha']))
        MSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan
        Var = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), X_unscaled.shape[1], K_fold*Nr)) * np.nan

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                # Rescaling to avoid validation dataset leakage
                scaler_x = StandardScaler(with_mean=True, with_std=True)
                scaler_x.fit(X_train)
                X_train_scale = scaler_x.transform(X_train)
                X_val_scale = scaler_x.transform(X_val)
                scaler_y = StandardScaler(with_mean=True, with_std=True)
                scaler_y.fit(y_train)
                y_train_scale = scaler_y.transform(y_train)
                y_val_scale = scaler_y.transform(y_val)
                temp = PAR(delayed(rm.EN_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[1], this_prod[0]) for this_prod in hyperparam_prod)
                _, Var[:, :, counter], _, MSE_result[:, counter], _, _ = zip(*temp)

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.nanargmin(MSE_mean)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            Var = np.sum(Var != 0, axis = 1) # Here, Var is n_hyperparams x n_features x n_folds
            Var_num = np.nansum(Var, axis = 1) # Here, Var is n_hyperparams x n_folds, and Var_num is n_hyperparams
            ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE

        # Hyperparameter setup
        l1_ratio, alpha = hyperparam_prod[ind]
        hyperparams = {'alpha': alpha, 'l1_ratio': l1_ratio}
        # Fit the final model
        if alpha:
            EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test = rm.EN_fitting(X, y, X_test, y_test, alpha, l1_ratio)
        else: # Alpha = 0 --> use ordinary least squares
            EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test = rm.OLS_fitting(X, y, X_test, y_test)
        return(hyperparams, EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'SPLS':
        if not(cv_type.startswith('Group')) and 'K' not in kwargs: # For non-grouped CV types
            kwargs['K'] = np.linspace( 1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), dtype = np.uint64)
        elif 'K' not in kwargs:
            kwargs['K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1), dtype = np.uint64)
        if 'eta' not in kwargs and model_name == 'SPLS':
            kwargs['eta'] = np.linspace(0, 1, 20, endpoint = False)[::-1]
        elif 'eta' not in kwargs and model_name == 'PLS': # Normal PLS is simply SPLS with eta = 0
            kwargs['eta'] = np.array([0])

        MSE_result = np.empty((len(kwargs['K']), len(kwargs['eta']), K_fold*Nr)) * np.nan
        if kwargs['robust_priority']:
            Var = np.empty((len(kwargs['K']), len(kwargs['eta']), K_fold*Nr)) * np.nan

        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
            # Rescaling to avoid validation dataset leakage
            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
            scaler_x_train.fit(X_train)
            X_train_scale = scaler_x_train.transform(X_train)
            X_val_scale = scaler_x_train.transform(X_val)
            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
            scaler_y_train.fit(y_train)
            y_train_scale = scaler_y_train.transform(y_train)
            y_val_scale = scaler_y_train.transform(y_val)
            for i in range(len(kwargs['K'])):
                for j in range(len(kwargs['eta'])):
                    _, variable, _, mse, _, _ = rm.SPLS_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, K = int(kwargs['K'][i]), eta = kwargs['eta'][j])
                    MSE_result[i, j, counter] = mse
                    if kwargs['robust_priority']:
                        Var[i, j, counter] = np.sum(variable.flatten() != 0)

        MSE_mean = np.nanmean(MSE_result, axis = 2)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(MSE_mean), MSE_mean.shape)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 2)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            Var_num = np.nansum(Var, axis = 2)
            ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0])

        # Hyperparameter setup
        K = int(kwargs['K'][ind[0]])
        eta = kwargs['eta'][ind[1]]
        hyperparams = {'K': int(K), 'eta': eta}
        # Fit the final model
        SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test = rm.SPLS_fitting(X, y, X_test, y_test, eta = eta, K = K)
        return(hyperparams, SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'PLS':  # TODO: merge with SPLS, since this is just a special case with eta = 0
        if not(cv_type.casefold().startswith('group')) and 'K' not in kwargs: # For non-grouped CV types
            kwargs['K'] = np.linspace(1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), dtype = int)
        elif 'K' not in kwargs:
            kwargs['K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1), dtype = int)

        MSE_result = np.zeros((len(kwargs['K']), K_fold*Nr))

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                # Rescaling to avoid validation dataset leakage
                scaler_x_train = StandardScaler(with_mean=True, with_std=True)
                scaler_x_train.fit(X_train)
                X_train_scale = scaler_x_train.transform(X_train)
                X_val_scale = scaler_x_train.transform(X_val)
                scaler_y_train = StandardScaler(with_mean=True, with_std=True)
                scaler_y_train.fit(y_train)
                y_train_scale = scaler_y_train.transform(y_train)
                y_val_scale = scaler_y_train.transform(y_val)
                temp = PAR(delayed(rm.PLS_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_K) for this_K in kwargs['K'])
                _, _, _, MSE_result[:, counter], _, _ = zip(*temp)

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.nanargmin(MSE_mean)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( kwargs['K'] == np.nanmin(kwargs['K'][MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE

        # Hyperparameter setup
        K = int(kwargs['K'][ind])
        hyperparams = {'K': K}

        # Fit the final model
        PLS_model = PLSRegression(K, scale = False, tol = eps).fit(X, y)
        PLS_params = PLS_model.coef_.squeeze()
        yhat_train = np.dot(X, PLS_params)
        yhat_test = np.dot(X_test, PLS_params)
        mse_train = MSE(yhat_train.flatten(), y.flatten())
        mse_test = MSE(yhat_test.flatten(), y_test.flatten())
        return(hyperparams, PLS_model, PLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'LCEN':
        kwargs['model_name'] = model_name # Sent to the joblib fun
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'label_name' not in kwargs: # Whether to auto-generate label names for the variables [x1, x2, ..., log(x1), ..., 1/x1, ..., x1*x2, etc.]
            kwargs['label_name'] = True # TODO: currently unused. Not sure whether I'll re-implement custom naming
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'all'
        if 'lag' not in kwargs:
            kwargs['lag'] = [0]
        if 'min_lag' not in kwargs:
            kwargs['min_lag'] = 0
        if 'LCEN_cutoff' not in kwargs:
            kwargs['LCEN_cutoff'] = 4e-3
        if 'LCEN_interaction' not in kwargs:
            kwargs['LCEN_interaction'] = True
        if 'LCEN_transform_y' not in kwargs:
            kwargs['LCEN_transform_y'] = False
        kwargs['selection'] = None # for the _LCEN_joblib_fun called below

        # First run for variable selection using a L1_ratio of 1 (that is, only using an L1 penalty)
        hyperparam_prod = list(product(kwargs['degree'], [1], kwargs['alpha'], kwargs['lag']))
        print(f'Beginning variable selection runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        with Parallel(n_jobs = -1) as PAR:
            if 'IC' in cv_type: # Information criterion
                temp = PAR(delayed(_LCEN_joblib_fun)(X, y, X_test, y_test, eps, kwargs, prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                temp = list(zip(*temp))[2] # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                temp = np.array(temp)
                if cv_type == 'AICc':
                    IC_result = temp[:, 1]
                elif cv_type == 'BIC':
                    IC_result = temp[:, 2]
                else: # AIC
                    IC_result = temp[:, 0]
                ind = np.argmin(IC_result)
            else: # Cross-validation
                MSE_result = np.empty((len(kwargs['degree']) * len(kwargs['alpha']) * len(kwargs['lag']), K_fold*Nr)) * np.nan
                for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                    temp = PAR(delayed(_LCEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs,
                            prod_idx, this_prod, counter) for prod_idx, this_prod in enumerate(hyperparam_prod))
                    MSE_result[:, counter], _, _ = zip(*temp)
                # Best hyperparameters for the preliminary run
                MSE_mean = np.nanmean(MSE_result, axis = 1)
                ind = np.nanargmin(MSE_mean)
        # Run to obtain the coefficients when LCEN is run with L1_ratio = 1
        degree, l1_ratio, alpha, lag = hyperparam_prod[ind]
        _, LCEN_params, _, _, _, _, label_names, _ = rm.LCEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, kwargs['min_lag'], tol = eps,
                                            trans_type = kwargs['trans_type'], interaction = kwargs['LCEN_interaction'], selection = None, transform_y = kwargs['LCEN_transform_y'])
        kwargs['selection'] = (np.abs(LCEN_params) >= kwargs['LCEN_cutoff'])&(np.abs(LCEN_params) != 0) # 1st clip step

        # Second run with a free L1_ratio but fixed degree and lag
        hyperparam_prod = list(product([degree], kwargs['l1_ratio'], kwargs['alpha'], [lag])) # Degree and lag have been fixed above
        print(f'Beginning real runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        with Parallel(n_jobs = -1) as PAR:
            if 'IC' in cv_type: # Information criterion
                temp = PAR(delayed(_LCEN_joblib_fun)(X, y, X_test, y_test, eps, kwargs, prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                temp = list(zip(*temp))[2] # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                temp = np.array(temp)
                if cv_type == 'AICc':
                    IC_result = temp[:, 1]
                elif cv_type == 'BIC':
                    IC_result = temp[:, 2]
                else: # AIC
                    IC_result = temp[:, 0]
                ind = np.argmin(IC_result)
            else:
                MSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan
                Var = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan # Used when robust_priority == True
                for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                    temp = PAR(delayed(_LCEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs, prod_idx,
                            this_prod, counter) for prod_idx, this_prod in enumerate(hyperparam_prod))
                    MSE_result[:, counter], Var[:, counter], _ = zip(*temp)
                # Best hyperparameters
                MSE_mean = np.nanmean(MSE_result, axis = 1)
                ind = np.nanargmin(MSE_mean)
                if kwargs['robust_priority']:
                    MSE_std = np.nanstd(MSE_result, axis = 1)
                    MSE_min = MSE_mean[ind]
                    MSE_bar = MSE_min + MSE_std[ind]
                    Var_num = np.nansum(Var, axis = 1)
                    ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
                    ind = ind[0][0]

        # Hyperparameter setup
        degree, l1_ratio, alpha, lag = hyperparam_prod[ind]
        hyperparams = {'degree': degree, 'l1_ratio': l1_ratio, 'alpha': alpha, 'lag': lag, 'cutoff': kwargs['LCEN_cutoff'], 'trans_type': kwargs['trans_type']}
        # Final run with the test set and best hyperparameters
        LCEN_model, LCEN_params, _, _, _, _, label_names, ICs = rm.LCEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, kwargs['min_lag'], tol = eps,
                                            trans_type = kwargs['trans_type'], interaction = kwargs['LCEN_interaction'], selection = kwargs['selection'], transform_y = kwargs['LCEN_transform_y'])
        label_names = label_names[kwargs['selection']]
        # Removing the features that had small coefficients after the final model selection (2nd clip step)
        final_selection = (np.abs(LCEN_params) >= kwargs['LCEN_cutoff'])&(np.abs(LCEN_params) != 0)
        LCEN_params = LCEN_params[final_selection].reshape(-1) # Reshape to avoid 0D arrays when only one variable is selected
        label_names = label_names[final_selection].reshape(-1)
        # Unscaling the model coefficients as per stackoverflow.com/questions/23642111/how-to-unscale-the-coefficients-from-an-lmer-model-fitted-with-a-scaled-respon
        if not kwargs['LCEN_transform_y'] and X.shape[1] > 0:
            X, X_test, _ = rm._feature_trans(X, X_test, degree, kwargs['LCEN_interaction'], kwargs['trans_type'])
        if lag > 0:
            X_temp = np.hstack([X[lag-1-idx : -idx-1, :] for idx in range(kwargs['min_lag'], lag)]) # The additional entries representing the previous times (t-1 to t-lag)
            y_temp = np.hstack([y[lag-1-idx : -idx-1] for idx in range(kwargs['min_lag'], lag)])
            X_test_temp = np.hstack([np.concatenate((X[X.shape[0]-idx-1:, :], X_test[: -idx-1, :])) for idx in range(kwargs['min_lag'], lag)]) # No need to remove entries from X_test or y_test because we can use the data from the final points of X or y to predict the initial points of X_test or y_test
            y_test_temp = np.hstack([np.concatenate((y[(len(y)-idx-1):],  y_test[: -idx-1])) for idx in range(kwargs['min_lag'], lag)])
            X = np.hstack((X[lag:], X_temp, y_temp))
            y = y[lag:] # Shorterning y
            X_test = np.hstack((X_test, X_test_temp, y_test_temp))
        if kwargs['LCEN_transform_y']: # Feature transformation that includes the y features in the X and X_test variables
            X, X_test, _ = rm._feature_trans(X, X_test, degree, kwargs['LCEN_interaction'], kwargs['trans_type'])
        LCEN_params_unscaled = np.zeros_like(LCEN_params)
        y_vars = np.array(['y' in label_names[idx] for idx in range(len(label_names))])
        if len(label_names): # if this is false, no variables were selected
            LCEN_params_unscaled[~y_vars] = (LCEN_params[~y_vars] * y.std() / X[:, kwargs['selection']][:, final_selection].std(axis=0)[~y_vars]) # Unscaling the X variables # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
            LCEN_params_unscaled[y_vars] = LCEN_params[y_vars]
        # Obtaining the predictions again since coefficients may have been removed
        yhat_train = np.dot(X[:, kwargs['selection']][:, final_selection], LCEN_params_unscaled) # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
        yhat_test = np.dot(X_test[:, kwargs['selection']][:, final_selection], LCEN_params_unscaled) # TODO: sometimes, when lag and min_lag > 0, an indexing problem occurs. Add [:, -len(kwargs['selection']):] before any X indexing to fix it
        intercept = (y.squeeze() - yhat_train).mean()
        if np.abs(intercept/y.mean()) >= kwargs['LCEN_cutoff']:
            yhat_train += intercept
            yhat_test += intercept
            label_names = np.concatenate((['intercept'], label_names))
            LCEN_params_unscaled = np.concatenate(([intercept], LCEN_params_unscaled))
        mse_train = MSE(y, yhat_train)
        mse_test = MSE(y_test, yhat_test)
        # Returning the results
        if kwargs['verbosity_level'] >= 3:
            print(f'{len(LCEN_params_unscaled) - int(label_names[0] == "intercept")} variables {"(and intercept)"*(label_names[0] == "intercept")} were selected' + ' '*20)
            print(f'The validation MSE was {MSE_mean[ind]:.3e}')
        LCEN_params_unscaled = list(zip(label_names, LCEN_params_unscaled))
        if 'IC' in cv_type:
            return(hyperparams, LCEN_model, LCEN_params_unscaled, mse_train, mse_test, yhat_train, yhat_test, IC_result[ind])
        else:
            return(hyperparams, LCEN_model, LCEN_params_unscaled, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'RF' or model_name == 'GBDT':
        if 'RF_n_estimators' not in kwargs:
            kwargs['RF_n_estimators'] = [10, 25, 50, 100, 200, 300]
        if 'RF_max_depth' not in kwargs:
            kwargs['RF_max_depth'] = [2, 3, 5, 10, 15, 20, 40]
        if 'RF_min_samples_leaf' not in kwargs:
            kwargs['RF_min_samples_leaf'] = [0.001, 0.01, 0.02, 0.05, 0.1]
        if 'RF_n_features' not in kwargs:
            kwargs['RF_n_features'] = [0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 1.0]
        # learning_rate matters only for GBDT -- set it to None to use RFs
        if model_name == 'RF':
            kwargs['learning_rate'] = [None]
        elif 'learning_rate' not in kwargs or len(kwargs['learning_rate']) == 0:
            kwargs['learning_rate'] = [0.01, 0.05, 0.1, 0.2]

        hyperparam_prod = list(product(kwargs['RF_n_estimators'], kwargs['RF_max_depth'], kwargs['RF_min_samples_leaf'], kwargs['RF_n_features'], kwargs['learning_rate']))
        MSE_result = np.empty( (len(kwargs['RF_n_estimators']) * len(kwargs['RF_max_depth']) * len(kwargs['RF_min_samples_leaf']) * len(kwargs['RF_n_features']) * len(kwargs['learning_rate']), K_fold*Nr) ) * np.nan
        """
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the parameters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['max_depth']), len(kwargs['n_estimators']), len(kwargs['min_samples_leaf'])))
            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        S[i, j, k] = i/len(kwargs['max_depth']) - k/len(kwargs['min_samples_leaf'])
        """

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                print(f'{model_name}: Beginning fold {counter+1}', end = '\r')
                # Rescaling to avoid validation dataset leakage
                scaler_x_train = StandardScaler(with_mean=True, with_std=True)
                scaler_x_train.fit(X_train)
                X_train_scale = scaler_x_train.transform(X_train)
                X_val_scale = scaler_x_train.transform(X_val)
                scaler_y_train = StandardScaler(with_mean=True, with_std=True)
                scaler_y_train.fit(y_train)
                y_train_scale = scaler_y_train.transform(y_train)
                y_val_scale = scaler_y_train.transform(y_val)
                temp = PAR(delayed(rm.forest_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1], this_prod[2], this_prod[3], this_prod[4]) for this_prod in hyperparam_prod)
                _, _, MSE_result[:, counter], _, _ = zip(*temp)

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.nanargmin(MSE_mean)
        """ # TODO: see above for the robust_priority changes
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( S == np.nanmin(S[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
        """

        # Hyperparameter setup
        n_estimators, max_depth, min_samples_leaf, n_features, learning_rate = hyperparam_prod[ind]
        if model_name == 'RF':
            hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'n_features': n_features}
        else: # learning_rate matters only for GBDT
            hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'n_features': n_features, 'learning_rate': learning_rate}
        # Fit the final model
        RF_model, mse_train, mse_test, yhat_train, yhat_test = rm.forest_fitting(X, y, X_test, y_test, n_estimators, max_depth, min_samples_leaf, n_features)
        return(hyperparams, RF_model, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'SVM':
        if 'SVM_gamma' not in kwargs or kwargs['SVM_gamma'] is None:
            gd = 1/X.shape[1]
            kwargs['SVM_gamma'] = [gd/50, gd/10, gd/5, gd/2, gd, gd*2, gd*5, gd*10, gd*50]
        if 'SVM_C' not in kwargs:
            kwargs['SVM_C'] = [0.01, 0.1, 1, 10, 50, 100]
        if 'SVM_epsilon' not in kwargs:
            kwargs['SVM_epsilon'] = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]

        hyperparam_prod = list(product(kwargs['SVM_gamma'], kwargs['SVM_C'], kwargs['SVM_epsilon']))
        MSE_result = np.empty( (len(kwargs['SVM_gamma']) * len(kwargs['SVM_C']) * len(kwargs['SVM_epsilon']), K_fold*Nr) ) * np.nan
        """
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the parameters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['C']), len(kwargs['gamma']), len(kwargs['epsilon'])))
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        S[i, j, k] = i/len(kwargs['C']) - j/len(kwargs['gamma']) - k/len(kwargs['epsilon'])
        """

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                print(f'{model_name}: Beginning fold {counter+1}', end = '\r')
                # Rescaling to avoid validation dataset leakage
                scaler_x_train = StandardScaler(with_mean=True, with_std=True)
                scaler_x_train.fit(X_train)
                X_train_scale = scaler_x_train.transform(X_train)
                X_val_scale = scaler_x_train.transform(X_val)
                scaler_y_train = StandardScaler(with_mean=True, with_std=True)
                scaler_y_train.fit(y_train)
                y_train_scale = scaler_y_train.transform(y_train)
                y_val_scale = scaler_y_train.transform(y_val)
                temp = PAR(delayed(rm.SVM_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1], this_prod[2]) for this_prod in hyperparam_prod)
                _, _, MSE_result[:, counter], _, _ = zip(*temp)

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.nanargmin(MSE_mean)
        """ # TODO: see above for the robust_priority changes
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( S == np.nanmin(S[MSE_mean < MSE_bar]) )[0][0] # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
        """

        # Hyperparameter setup
        gamma, C, epsilon = hyperparam_prod[ind]
        hyperparams = {'gamma': gamma, 'C': C, 'epsilon': epsilon}
        # Fit the final model
        SVM_model, mse_train, mse_test, yhat_train, yhat_test = rm.SVM_fitting(X, y, X_test, y_test, gamma, C, epsilon)
        return(hyperparams, SVM_model, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'AdaB':
        if 'RF_n_estimators' not in kwargs:
            kwargs['RF_n_estimators'] = [10, 25, 50, 100, 200, 300]
        if 'learning_rate' not in kwargs or len(kwargs['learning_rate']) == 0:
            kwargs['learning_rate'] = [0.01, 0.05, 0.1, 0.2]

        hyperparam_prod = list(product(kwargs['RF_n_estimators'], kwargs['learning_rate']))
        MSE_result = np.empty( (len(kwargs['RF_n_estimators']) * len(kwargs['learning_rate']), K_fold*Nr) ) * np.nan

        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                # Rescaling to avoid validation dataset leakage
                scaler_x_train = StandardScaler(with_mean=True, with_std=True)
                scaler_x_train.fit(X_train)
                X_train_scale = scaler_x_train.transform(X_train)
                X_val_scale = scaler_x_train.transform(X_val)
                scaler_y_train = StandardScaler(with_mean=True, with_std=True)
                scaler_y_train.fit(y_train)
                y_train_scale = scaler_y_train.transform(y_train)
                y_val_scale = scaler_y_train.transform(y_val)
                temp = PAR(delayed(rm.AdaBoost_fitting)(X_train_scale, y_train_scale, X_val_scale, y_val_scale, this_prod[0], this_prod[1]) for this_prod in hyperparam_prod)
                _, _, MSE_result[:, counter], _, _ = zip(*temp)

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.nanargmin(MSE_mean)

        # Hyperparameter setup
        n_estimators, learning_rate = hyperparam_prod[ind]
        hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
        # Fit the final model
        AdaB_model, mse_train, mse_test, yhat_train, yhat_test = rm.AdaBoost_fitting(X, y, X_test, y_test, n_estimators, learning_rate)
        return(hyperparams, AdaB_model, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name in {'MLP', 'RNN'}:
        # Loss function
        kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'use_cross_entropy' not in kwargs or not kwargs['use_cross_entropy']:
            loss_function = torch.nn.functional.mse_loss
        else:
            if 'class_weight' not in kwargs or kwargs['class_weight'] is None:
                kwargs['class_weight'] = torch.ones(np.max(y) + int(0 in y)) # If 0 also represents a class, then there are max(y) + 1 classes
            loss_function = torch.nn.CrossEntropyLoss(weight = torch.Tensor(kwargs['class_weight'])).to(kwargs['device'])
        # Layer hyperparameters
        if kwargs['use_cross_entropy']:
            myshape_y = np.max(y) + int(0 in y) # TODO: len(set(y)) should work better, especially if one of the classes is missing for some reason
        else:
            myshape_y = y.shape[1]
        if 'MLP_layers' not in kwargs or kwargs['MLP_layers'] is None:
            myshape_X = X.shape[1]
            kwargs['MLP_layers'] = [[(myshape_X, myshape_X*2), (myshape_X*2, myshape_y)], [(myshape_X, myshape_X), (myshape_X, myshape_y)], [(myshape_X, myshape_X//2), (myshape_X//2, myshape_y)], # One hidden layer
                                    [(myshape_X, myshape_X*2), (myshape_X*2, myshape_X*2), (myshape_X*2, myshape_y)], [(myshape_X, myshape_X*2), (myshape_X*2, myshape_X), (myshape_X, myshape_y)], # Two hidden layers
                                    [(myshape_X, myshape_X), (myshape_X, myshape_X), (myshape_X, myshape_y)], [(myshape_X, myshape_X), (myshape_X, myshape_X//2), (myshape_X//2, myshape_y)] ]
        if model_name == 'RNN':
            if 'RNN_layers' not in kwargs or kwargs['RNN_layers'] is None:
                kwargs['RNN_layers'] = X.shape[1]
            # Ensuring the MLP layers are compatible with the RNNs by making the MLP layers' first size equal to the output size of the last RNN
            last_lstm_size = kwargs['RNN_layers'] if isinstance(kwargs['RNN_layers'], int) else kwargs['RNN_layers'][-1] # For convenience, since kwargs['RNN_layers'] could be a single int or a list
            for cur_layer in kwargs['MLP_layers']:
                cur_layer[0] = (last_lstm_size, cur_layer[0][1])
        elif model_name == 'MLP':
            kwargs['RNN_layers'] = 0
        # Other model and training hyperparameters
        if 'activation' not in kwargs:
            kwargs['activation'] = ['relu']
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = [1e-2, 5e-3]
        if 'weight_decay' not in kwargs:
            kwargs['weight_decay'] = 0
        if 'l1_penalty_factor' not in kwargs:
            kwargs['l1_penalty_factor'] = 0
        if 'n_epochs' not in kwargs:
            kwargs['n_epochs'] = 100
        # Scheduler hyperparameters
        if 'scheduler' not in kwargs:
            kwargs['scheduler'] = 'plateau'
        if 'scheduler_mode' not in kwargs:
            kwargs['scheduler_mode'] = 'min'
        if 'scheduler_factor' not in kwargs:
            kwargs['scheduler_factor'] = 0.5
        if 'scheduler_patience' not in kwargs:
            kwargs['scheduler_patience'] = 10
        if 'scheduler_min_lr' not in kwargs:
            kwargs['scheduler_min_lr'] = 1/16
        if 'scheduler_last_epoch' not in kwargs or kwargs['scheduler_last_epoch'] is None:
            kwargs['scheduler_last_epoch'] = kwargs['n_epochs'] - 30
        elif kwargs['scheduler_last_epoch'] < 0:
            kwargs['scheduler_last_epoch'] = kwargs['n_epochs'] - kwargs['scheduler_last_epoch']
        if 'scheduler_warmup' not in kwargs:
            kwargs['scheduler_warmup'] = 10
        if 'val_loss_file' not in kwargs or kwargs['val_loss_file'] is None:
            time_now = '-'.join([str(elem) for elem in localtime()[:6]]) # YYYY-MM-DD-hh-mm-ss
            kwargs['val_loss_file'] = f'ANN_val-loss_{time_now}.csv'
        if 'expand_hyperparameter_search' not in kwargs:
            kwargs['expand_hyperparameter_search'] = False

        def CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs, hyperparam_list = None):
            """
            This function runs a cross-validation procedure for each combination of MLP / RNN hyperparameters.
            Results are saved in the kwargs['val_loss_file'] .csv file.
            """
            # Recording the validation losses
            try:
                final_val_loss = pd.read_csv(kwargs['val_loss_file'], index_col = [0, 1])
            except FileNotFoundError:
                print('NOTE: no validation loss file was found, so the cross-validation will begin from the first set of hyperparameters.')
                my_prod = list(product(kwargs['activation'], kwargs['learning_rate']))
                my_idx = pd.MultiIndex.from_tuples(my_prod)
                final_val_loss = pd.DataFrame(np.nan, index = my_idx, columns = [str(elem) for elem in kwargs['MLP_layers']])

            # Train and validate
            if hyperparam_list is None:
                hyperparam_list = list(product(kwargs['MLP_layers'], kwargs['learning_rate'], kwargs['activation']))
            for cur_idx, cur_hp in enumerate(hyperparam_list): # cur_hp is (layers, lr, activation)
                # We added a new layer configuration to the hyperparameters
                if not str(cur_hp[0]) in list(final_val_loss.columns):
                    final_val_loss.insert(kwargs['MLP_layers'].index(cur_hp[0]), str(cur_hp[0]), np.nan) # layers.index to ensure consistent order
                # We added a new activation or learning rate to the hyperparameters
                if not (cur_hp[2], cur_hp[1]) in final_val_loss.index.to_list():
                    final_val_loss.loc[(cur_hp[2], cur_hp[1]), :] = np.nan
                    final_val_loss = final_val_loss.sort_index(ascending = [True, False]) # Sorting the indices

                # Run CV only if we do not have validation losses for this set of parameters
                if np.isnan( final_val_loss.at[(cur_hp[2], cur_hp[1]), str(cur_hp[0])] ):
                    if kwargs['verbosity_level'] >= 2:
                        print(f'Beginning hyperparameters {cur_idx+1:3}/{len(hyperparam_list)}: MLP layers = {cur_hp[0]}; LR = {cur_hp[1]}; activation = {cur_hp[2]}   ', end = '\r')
                    scheduler_min_lr = kwargs['scheduler_min_lr'] * cur_hp[1]
                    temp_val_loss = 0
                    for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                        if kwargs['verbosity_level'] >= 3:
                            print(f'Current fold: {counter+1}', end = '\r')
                        # Rescaling to avoid validation dataset leakage
                        if kwargs['scale_X'] and len(X.shape) == 2: # StandardScaler doesn't work with 3D arrays
                            scaler_x_train = StandardScaler(with_mean=True, with_std=True)
                            scaler_x_train.fit(X_train)
                            X_train_scale = torch.Tensor(scaler_x_train.transform(X_train))
                            X_val_scale = torch.Tensor(scaler_x_train.transform(X_val))
                        else:
                            X_train_scale = torch.Tensor(X_train)
                            X_val_scale = torch.Tensor(X_val)
                        if kwargs['scale_y'] and not kwargs['use_cross_entropy']:
                            scaler_y_train = StandardScaler(with_mean=True, with_std=True)
                            scaler_y_train.fit(y_train)
                            y_train_scale = torch.Tensor(scaler_y_train.transform(y_train))
                            y_val_scale = torch.Tensor(scaler_y_train.transform(y_val))
                        elif kwargs['use_cross_entropy']:
                            y_train_scale = torch.LongTensor(y_train)
                            y_val_scale = torch.LongTensor(y_val)
                        else:
                            y_train_scale = torch.Tensor(y_train)
                            y_val_scale = torch.Tensor(y_val)
                        # Creating the Datasets / DataLoaders + run setup
                        train_dataset_fold = MyDataset(X_train_scale, y_train_scale)
                        train_loader_fold = DataLoader(train_dataset_fold, kwargs['batch_size'], shuffle = True)
                        val_dataset_fold = MyDataset(X_val_scale, y_val_scale)
                        val_loader_fold = DataLoader(val_dataset_fold, kwargs['batch_size'], shuffle = True)
                        if not kwargs['use_cross_entropy']: # Regression
                            best_metric_fold = 1000 * np.maximum(1, y_train_scale.max().item())
                        else:
                            best_metric_fold = 0
                        epochs_with_improvement = []
                        epochs_with_improvement_print = []
                        # Declaring the model and optimizer
                        model = my_ANN(cur_hp[0], cur_hp[2], X_train_scale.shape[-1], kwargs['RNN_layers'], kwargs['device']).to(kwargs['device'])
                        optimizer = torch.optim.AdamW(model.parameters(), lr = cur_hp[1], weight_decay = kwargs['weight_decay'])
                        if kwargs['scheduler'].casefold() == 'plateau':
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, kwargs['scheduler_mode'], kwargs['scheduler_factor'], kwargs['scheduler_patience'], min_lr = scheduler_min_lr)
                        elif kwargs['scheduler'].casefold() == 'cosine':
                            scheduler = CosineScheduler(kwargs['scheduler_last_epoch'], cur_hp[1], warmup_steps = kwargs['scheduler_warmup'], final_lr = scheduler_min_lr)
                        elif kwargs['scheduler'].casefold() == 'lambda':
                            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'step':
                            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, kwargs['scheduler_patience'], kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'multistep':
                            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, kwargs['scheduler_patience'], kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'exponential':
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, kwargs['scheduler_factor'])
                        for epoch in range(kwargs['n_epochs']):
                            if kwargs['verbosity_level'] >= 4:
                                print(f'Fold {counter+1}; epoch {epoch+1:2}/{kwargs["n_epochs"]}; Best Metric = {best_metric_fold:5.2f}; Imp: {epochs_with_improvement_print}', end = '\r')
                            train_loss, _ = loop_model(model, optimizer, train_loader_fold, loss_function, epoch, kwargs['batch_size'], categorical = kwargs['use_cross_entropy'], l1_penalty_factor = kwargs['l1_penalty_factor'])
                            val_loss, _ = loop_model(model, optimizer, val_loader_fold, loss_function, epoch, kwargs['batch_size'], evaluation = True, categorical = kwargs['use_cross_entropy'], l1_penalty_factor = kwargs['l1_penalty_factor'])
                            if (not kwargs['use_cross_entropy'] and val_loss < best_metric_fold) or (kwargs['use_cross_entropy'] and val_loss > best_metric_fold):
                                best_metric_fold = val_loss
                                epochs_with_improvement.append(epoch+1)
                                epochs_with_improvement_print.append(f'{epoch+1}:{best_metric_fold:.1e}')
                            if 'scheduler' in locals() and scheduler.__module__ == 'torch.optim.lr_scheduler': # Pytorch built-in scheduler
                                scheduler.step(val_loss)
                            elif 'scheduler' in locals():
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = scheduler(epoch)
                        # Recording the validation loss for this fold
                        if kwargs['verbosity_level'] >= 3:
                            print(f'Fold {counter+1} done; Best Loss = {best_metric_fold:5.2f}; Improvement = {epochs_with_improvement}' + ' '*18)
                        temp_val_loss += best_metric_fold
                    # Saving the average validation loss after CV
                    final_val_loss.at[(cur_hp[2], cur_hp[1]), str(cur_hp[0])] = temp_val_loss / (counter+1) # counter+1 because the correct value may be K_fold (for simple Kfold) or K_fold*Nr (for repeated K_fold)
                    final_val_loss.to_csv(kwargs['val_loss_file'])
            return final_val_loss
        final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs)
        # Finding the best hyperparameters
        best_idx, best_LR, best_neurons, best_act, best_act_loc = _get_best_hyperparameters(final_val_loss)
        # Checking whether the best hyperparameters are in the extremes of what was cross-validated
        extreme_LR = best_idx[0] in {best_act_loc.start, best_act_loc.stop-1} # Whether the best LR was either the highest or the lowest value checked for the best activation function
        layer_mask = np.array([len(elem) for elem in kwargs['MLP_layers']]) == len(best_neurons) # Check only MLP configurations with the same number of layers
        layers_for_extreme = sorted([elem for idx, elem in enumerate(kwargs['MLP_layers']) if layer_mask[idx]])
        if len(layers_for_extreme) >= 2:
            extreme_neuron = best_neurons in [layers_for_extreme[0], layers_for_extreme[-1]] # Whether the best neuron was either the largest or the smallest combination checked # TODO: MLP configurations with more than 1 hidden layer will benefit from a more thorough analysis of extremity
        else:
            extreme_neuron = False
        while (extreme_LR or extreme_neuron) and kwargs['expand_hyperparameter_search']:
            print(f'Expanding the hyperparameters using the "{kwargs["expand_hyperparameter_search"]}" rule')
            hyperparam_list = list(product(kwargs['MLP_layers'], kwargs['learning_rate'], kwargs['activation'])) # Used when kwargs['expand_hyperparameter_search'] == 'single'
            # New LR value
            if best_idx[0] == best_act_loc.start: # The best LR was the highest value tested
                new_LR = best_LR * np.sqrt(10)
                kwargs['learning_rate'].append(new_LR)
            elif best_idx[0] == best_act_loc.stop-1: # The best LR was the lowest value tested
                new_LR = best_LR / np.sqrt(10)
                kwargs['learning_rate'].append(new_LR)
            # New MLP layer configuration
            new_neurons = best_neurons[:] # [:] makes a copy
            if best_neurons == layers_for_extreme[0] and best_neurons[0][1] != 1: # The best layer configuration was the smallest configuration tested. != 1 to avoid infinite loop
                if best_neurons[0][1] > X.shape[1]:
                    new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] - X.shape[1])
                    new_neurons[1] = (new_neurons[1][0] - X.shape[1], new_neurons[0][0])
                else: # The layer is <= X.shape[1], so we'll simply divide it by 2
                    new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] // 2)
                    new_neurons[1] = (new_neurons[1][0] // 2, new_neurons[0][0])
                kwargs['MLP_layers'].append(new_neurons)
                if len(best_neurons) > 2: # 2 (or more) hidden layers TODO
                    pass
            elif best_neurons == layers_for_extreme[-1]: # The best layer configuration was the largest configuration tested
                new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] + X.shape[1])
                new_neurons[1] = (new_neurons[1][0] + X.shape[1], new_neurons[0][0])
                kwargs['MLP_layers'].append(new_neurons)
                if len(best_neurons) > 2: # 2 (or more) hidden layers TODO
                    pass
            if kwargs['expand_hyperparameter_search'].casefold() == 'grid':
                if best_act not in kwargs['activation']:
                    print(f'WARNING: the best activation function ({best_act}) is not among the input activations, which means hyperparameters using it will not be tested\n' +
                          f'\tInclude it in the SPA call (as activation = ["{best_act}", "other", "activations"]) to ensure its hyperparameters, too, will be tested.')
                final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs)
            elif kwargs['expand_hyperparameter_search'].casefold() == 'single':
                if 'new_LR' in locals() and 'new_neurons' in locals() and (new_neurons, new_LR, best_act) not in hyperparam_list: # Adding both a new LR and a new layer configuration
                    hyperparam_list.append((new_neurons, new_LR, best_act))
                if 'new_LR' in locals() and (best_neurons, new_LR, best_act) not in hyperparam_list: # Adding a new LR with the other best hyperparameters
                    hyperparam_list.append((best_neurons, new_LR, best_act))
                    del new_LR # To avoid an infinite loop, as the if statement check for the presence of this variable in locals()
                if 'new_neurons' in locals() and (new_neurons, best_LR, best_act) not in hyperparam_list: # Adding a new layer configuration with the other best hyperparameters
                    hyperparam_list.append((new_neurons, best_LR, best_act))
                    del new_neurons # To avoid an infinite loop, as the if statement check for the presence of this variable in locals()
                final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs, hyperparam_list)
            # Finding the best hyperparameters
            best_idx, best_LR, best_neurons, best_act, best_act_loc = _get_best_hyperparameters(final_val_loss)
            # Checking whether the best hyperparameters are in the extremes of what was cross-validated
            extreme_LR = best_idx[0] in {best_act_loc.start, best_act_loc.stop-1} # Whether the best LR was either the highest or the lowest value checked for the best activation function
            layer_mask = np.array([len(elem) for elem in kwargs['MLP_layers']]) == len(best_neurons) # Check only MLP configurations with the same number of layers
            layers_for_extreme = np.sort(np.array(kwargs['MLP_layers'], dtype = object)[layer_mask])
            extreme_neuron = (best_neurons in [layers_for_extreme[0], layers_for_extreme[-1]] and best_neurons[0][1] != 1) # Whether the best neuron was either the largest or the smallest combination checked # TODO: MLP configurations with more than 1 hidden layer will benefit from a more thorough analysis of extremity
        # Creating LongTensors if using cross-entropy
        if not kwargs['use_cross_entropy']:
            y = torch.Tensor(y)
            y_test = torch.Tensor(y_test)
        else:
            y = torch.LongTensor(y)
            y_test = torch.LongTensor(y_test)
        # Final model training
        train_dataset = MyDataset(torch.Tensor(X), y)
        train_loader = DataLoader(train_dataset, kwargs['batch_size'], shuffle = True)
        # Re-declaring the model
        model = my_ANN(best_neurons, best_act, X.shape[-1], kwargs['RNN_layers'], kwargs['device']).to(kwargs['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr = best_LR, weight_decay = kwargs['weight_decay'])
        if kwargs['scheduler'].casefold() in {'plateau', 'cosine'}: # The plateau scheduler cannot work here because there is no validation loss, so it gets switched for the cosine scheduler
            scheduler = CosineScheduler(kwargs['scheduler_last_epoch'], base_lr = best_LR, warmup_steps = kwargs['scheduler_warmup'], final_lr = best_LR * kwargs['scheduler_min_lr'])
        # Retrain
        if kwargs['verbosity_level'] >= 2: print('Beginning final ANN training')
        for epoch in range(kwargs['n_epochs']):
            loop_model(model, optimizer, train_loader, loss_function, epoch, kwargs['batch_size'], categorical = kwargs['use_cross_entropy'], l1_penalty_factor = kwargs['l1_penalty_factor'])
            if 'scheduler' in locals() and scheduler.__module__ == 'torch.optim.lr_scheduler': # Pytorch built-in scheduler
                scheduler.step() # TODO: we do not really have a val_loss here. Need to check how the other built-in Schedulers behave
            elif 'scheduler' in locals():
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)
        # Final evaluation - Train set
        train_dataset = MyDataset(torch.Tensor(X), y)
        train_loader = DataLoader(train_dataset, kwargs['batch_size'], shuffle = False) # Not shuffling because this is just for evaluation
        model.eval()
        train_pred = torch.empty((len(train_loader.dataset), myshape_y))
        if not kwargs['use_cross_entropy']:
            train_y = torch.empty(len(train_loader.dataset))
        else:
            train_y = torch.empty(len(train_loader.dataset), dtype = torch.long)
        for idx, data in enumerate(train_loader):
            X_finaleval, y_finaleval = data # _finaleval to avoid overwriting the original X and y
            X_finaleval = X_finaleval.cuda()
            pred = model(X_finaleval).cpu().detach()
            train_pred[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(pred), :] = pred
            train_y[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(y_finaleval)] = y_finaleval.squeeze()
        train_loss = loss_function(train_pred.squeeze(), train_y).item()
        # Final evaluation - Test set
        test_dataset = MyDataset(torch.Tensor(X_test), y_test)
        test_loader = DataLoader(test_dataset, kwargs['batch_size'], shuffle = False) # Not shuffling because this is just for evaluation
        test_pred = torch.empty((len(test_loader.dataset), myshape_y))
        if not kwargs['use_cross_entropy']:
            test_y = torch.empty(len(test_loader.dataset))
        else:
            test_y = torch.empty(len(test_loader.dataset), dtype = torch.long)
        for idx, data in enumerate(test_loader):
            X_finaleval, y_finaleval = data # _finaleval to avoid overwriting the original X and y
            X_finaleval = X_finaleval.cuda()
            pred = model(X_finaleval).cpu().detach()
            test_pred[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(pred), :] = pred
            test_y[idx*kwargs['batch_size']:(idx*kwargs['batch_size'])+len(y_finaleval)] = y_finaleval.squeeze()
        test_loss = loss_function(test_pred.squeeze(), test_y).item()
        best_hyperparameters = {'RNN size': kwargs['RNN_layers'], 'MLP size': str(best_neurons), 'LR': best_LR, 'activation': best_act, 'batch_size': kwargs['batch_size'], 'weight_decay': kwargs['weight_decay'], 'scheduler': kwargs['scheduler']}
        if kwargs['scheduler'].casefold() in {'plateau', 'lambda', 'step', 'multistep', 'exponential'}:
            best_hyperparameters['scheduler_factor'] = kwargs['scheduler_factor']
        if kwargs['scheduler'].casefold() in {'plateau', 'step', 'multistep'}:
            best_hyperparameters['scheduler_patience'] = kwargs['scheduler_patience']
        if kwargs['scheduler'].casefold() == 'cosine':
            best_hyperparameters['scheduler_warmup'] = kwargs['scheduler_warmup']
            best_hyperparameters['scheduler_last_epoch'] = kwargs['scheduler_last_epoch']
            best_hyperparameters['scheduler_min_lr'] = kwargs['scheduler_min_lr']
        return model, final_val_loss, train_loss, test_loss, np.array(train_pred, dtype = float), np.array(test_pred, dtype = float), best_hyperparameters # Converting to float to save as JSON in SPA.py

@ignore_warnings()
def _LCEN_joblib_fun(X_train, y_train, X_val, y_val, eps, kwargs, prod_idx, this_prod, counter = -1):
    """
    A helper function to parallelize LCEN. Shouldn't be called by the user
    """
    degree, l1_ratio, alpha, lag = this_prod
    if (prod_idx == 0 or not (prod_idx+1)%100) and counter >= 0: # CV
        print(f'Beginning run {prod_idx+1:4} of fold {counter+1:3}', end = '\r')
    elif prod_idx == 0 or not (prod_idx+1)%100: # IC -- no folds
        print(f'Beginning run {prod_idx+1:4}', end = '\r')
    _, variable, _, mse, _, _, _, ICs = rm.LCEN_fitting(X_train, y_train, X_val, y_val, alpha, l1_ratio, degree, lag, kwargs['min_lag'], tol = eps,
                            trans_type = kwargs['trans_type'], interaction = kwargs['LCEN_interaction'], selection = kwargs['selection'], transform_y = kwargs['LCEN_transform_y'])
    return mse, np.sum(variable.flatten() != 0), ICs

class MyDataset(Dataset):
    def __init__(self, Xdata, ydata):
        self.Xdata = Xdata
        self.ydata = ydata

    def __len__(self):
        return len(self.Xdata)

    def __getitem__(self, idx):
        return self.Xdata[idx], self.ydata[idx]

class CosineScheduler: # For MLPs and RNNs. Code obtained from https://d2l.ai/chapter_optimization/lr-scheduler.html
    def __init__(self, max_update, base_lr = 0.01, final_lr = 0, warmup_steps = 0, warmup_begin_lr = 0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + np.cos(
                np.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

# A helper function that is called every epoch of training or validation for MLPs and RNNs
def loop_model(model, optimizer, loader, loss_function, epoch, batch_size, evaluation = False, categorical = False, l1_penalty_factor = 0):
    if evaluation:
        model.eval()
        myshape_y = len(set(loader.dataset.ydata.squeeze(1).tolist())) # TODO: confirm that squeeze(1) is the right operation (and not squeeze(0))
        val_pred = torch.empty((len(loader.dataset), myshape_y))
        val_y = torch.empty((len(loader.dataset)), dtype = torch.long)
    else:
        model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_loss = 0
    for idx, data in enumerate(loader):
        X, y = data
        X = X.to(device)
        y = y.squeeze(1).to(device)
        pred = model(X, categorical)
        if 'total_pred_y' not in locals():
            total_pred_y = torch.empty((len(loader.dataset), pred.shape[1]))
            if len(y.shape) > 1:
                real_y = torch.empty((len(loader.dataset), y.shape[1]))
            else:
                real_y = torch.empty(len(loader.dataset))
        total_pred_y[idx*batch_size:(idx*batch_size)+len(pred), :] = pred.cpu().detach()
        if len(y.shape) > 1:
            real_y[idx*batch_size:(idx*batch_size)+len(y), :] = y.cpu().detach()
        else:
            real_y[idx*batch_size:(idx*batch_size)+len(y)] = y.cpu().detach()
        # Group L1 penalty
        l1_penalty = 0
        if l1_penalty_factor:
            params = model.named_parameters()
            for parameter_name, parameter in params:
                if 'weight' in parameter_name:
                    l1_penalty += torch.sum( np.sqrt(parameter.shape[0]) * torch.sqrt(torch.sum(parameter**2, axis=0)) )
        loss = loss_function(pred.squeeze(1), y) + l1_penalty_factor * l1_penalty
        total_loss += y.shape[0]*loss.item()/loader.dataset.ydata.shape[0]
        # Backpropagation
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif categorical:
            val_pred[idx*batch_size:(idx*batch_size)+len(pred), :] = pred.cpu().detach()
            val_y[idx*batch_size:(idx*batch_size)+len(y)] = y
    if evaluation and categorical:
        val_pred_CM = val_pred.argmax(axis=1)
        CM = confusion_matrix(val_y, val_pred_CM) # Confusion matrix to make F1 calcs easier
        if CM[1,1]+CM[1,0] and CM[1,1]+CM[0,1]: # Avoids dividing by 0
            rec = CM[1,1]/(CM[1,1]+CM[1,0])
            pre = CM[1,1]/(CM[1,1]+CM[0,1])
        else:
            rec, pre = 0, 0
        if rec and pre: # Avoids dividing by 0 when calculating F1
            total_loss = -2/(1/rec + 1/pre) # Calling it "total_loss" for convienience, but this is just the F1 score
        else:
            total_loss = 0
    return total_loss, total_pred_y

# MLP or LSTM+MLP model
class my_ANN(torch.nn.Module):
    def __init__(self, layers, activ_fun = 'relu', lstm_input_size = 0, lstm_hidden_size = 0, device = 'cuda'):
        super(my_ANN, self).__init__()
        # Setup to convert string to activation function
        if activ_fun == 'relu':
            torch_activ_fun = torch.nn.ReLU()
        elif activ_fun == 'tanh':
            torch_activ_fun = torch.nn.Tanh()
        elif activ_fun == 'sigmoid':
            torch_activ_fun = torch.nn.Sigmoid()
        elif activ_fun == 'tanhshrink':
            torch_activ_fun = torch.nn.Tanhshrink()
        elif activ_fun == 'selu':
            torch_activ_fun = torch.nn.SELU()
        else:
            raise ValueError(f'Invalid activ_fun. You passed {activ_fun}')
        # LSTM cell
        if isinstance(lstm_hidden_size, int) and lstm_hidden_size:
            self.lstm = [torch.nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first = True, bidirectional = True).to(device)] # Need to send to device because the cells are in a list
        elif isinstance(lstm_hidden_size, (list, tuple)):
            self.lstm = []
            for idx, size in enumerate(lstm_hidden_size):
                if idx == 0:
                    self.lstm.append(torch.nn.LSTM(lstm_input_size, size, batch_first = True, bidirectional = True).to(device)) # Need to send to device because the cells are in a list
                else:
                    self.lstm.append(torch.nn.LSTM(lstm_hidden_size[idx-1], size, batch_first = True, bidirectional = True).to(device)) # Need to send to device because the cells are in a list
        if 'lstm' in dir(self):
            self.lstm = torch.nn.ModuleList(self.lstm) # Need to transform list into a ModuleList so PyTorch updates and interacts with the weights properly
        # Transforming layers list into OrderedDict with layers + activation
        mylist = list()
        for idx, elem in enumerate(layers):
            mylist.append((f'Linear{idx}', torch.nn.Linear(layers[idx][0], layers[idx][1]) ))
            if idx < len(layers)-1:
                mylist.append((f'{activ_fun}{idx}', torch_activ_fun))
        # OrderedDict into NN
        self.model = torch.nn.Sequential(OrderedDict(mylist))
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self) -> None: # Adapted from from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L105 and https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py#L1191
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for weight in self.parameters():
            if len(weight.shape) > 1:
                torch.nn.init.kaiming_uniform_(weight, a = np.sqrt(5), nonlinearity = 'relu')

    def forward(self, x, categorical = False):
        if 'lstm' in dir(self):
            for cell in self.lstm:
                x, (ht, _) = cell(x)
                if cell.bidirectional:
                    x = (x[:, :, :x.shape[2]//2] + x[:, :, x.shape[2]//2:]) / 2 # Average between forward and backward
            if cell.bidirectional:
                to_MLP = (ht[0] + ht[1]) / 2 # Average between forward and backward
            else:
                to_MLP = ht
            out = self.model(to_MLP)
        else:
            out = self.model(x)
        if categorical:
            return self.sigmoid(out)
        else:
            return out

def _get_best_hyperparameters(final_val_loss):
    """
    A helper function to obtain and format the best hyperparameters after cross-validation of an MLP or RNN model.
    Is called automatically by SPA and shouldn't be called by the user.
    """
    best_idx = np.unravel_index(np.nanargmin(final_val_loss.values), final_val_loss.shape)
    best_LR = final_val_loss.index[best_idx[0]][1]
    best_neurons_str = final_val_loss.columns[best_idx[1]]
    best_act = final_val_loss.index[best_idx[0]][0]
    best_act_loc = final_val_loss.index.get_loc(best_act)
    # Converting the best number of neurons from str to list
    best_neurons = []
    temp_number = []
    temp_tuple = []
    for elem in best_neurons_str:
        if elem in '0123456789':
            temp_number.append(elem)
        elif elem in {',', ')'} and temp_number: # Finished a number. 2nd check because there is a comma right after )
            converted_number = ''.join(temp_number)
            temp_tuple.append( int(converted_number) )
            temp_number = []
        if elem in {')'}: # Also finished a tuple
            best_neurons.append(tuple(temp_tuple))
            temp_tuple = []
    return best_idx, best_LR, best_neurons, best_act, best_act_loc
