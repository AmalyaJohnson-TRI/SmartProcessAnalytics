"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import ignore_warnings
from sklearn.metrics import mean_squared_error as MSE
import regression_models as rm
import nonlinear_regression_other as nro
from itertools import product
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from itertools import product
from collections import OrderedDict
from time import localtime
import pdb

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
        Used when model_name not in {'ALVEN', 'DALVEN'}, as these already use unscaled inputs
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
        kwargs['l1_ratio'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
    if 'alpha' not in kwargs: # Unusual scenario, since SPA passes a default kwargs['alpha'] == 20
        kwargs['alpha'] = np.concatenate(([0], np.logspace(-4.3, 0, 20)))
    elif isinstance(kwargs['alpha'], int): # User passed an integer instead of a list of values
        kwargs['alpha'] = np.concatenate(([0], np.logspace(-4.3, 0, kwargs['alpha'])))
    if 'use_cross_entropy' not in kwargs:
        kwargs['use_cross_entropy'] = False

    if model_name == 'EN':
        MSE_result = np.empty((len(kwargs['alpha']), len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan
        if kwargs['robust_priority']:
            Var = np.empty((len(kwargs['alpha']), len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan

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
            for j in range(len(kwargs['l1_ratio'])):
                for i in range(len(kwargs['alpha'])):
                    _, variable, _, mse, _, _ = rm.EN_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, kwargs['alpha'][i], kwargs['l1_ratio'][j])
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
        alpha = kwargs['alpha'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[1]]
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
        if 'eta' not in kwargs:
            kwargs['eta'] = np.linspace(0, 1, 20, endpoint = False)[::-1] #eta = 0 -> use normal PLS

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

    elif model_name == 'POLY': # TODO: add onestd? # TODO: POLY can't even be called from SPA()
        if 'degree' not in kwargs:
            kwargs['degree'] = [2,3,4]
        if 'interaction' not in kwargs:
            kwargs['interaction'] = True
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'poly'

        MSE_result = np.empty((len(kwargs['degree']), K_fold*Nr)) * np.nan

        for d in range(len(kwargs['degree'])):
            X_trans, _ = rm._feature_trans(X, X_test, kwargs['degree'][d], kwargs['interaction'], kwargs['trans_type'])
            X_trans = np.hstack((np.ones([X_trans.shape[0], 1]), X_trans))
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_trans, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                _, _, _, mse, _, _ = rm.OLS_fitting(X_train, y_train, X_val, y_val)
                MSE_result[d, counter] = mse

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(MSE_mean), MSE_mean.shape)
        """if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            Var_num = np.nansum(Var, axis = 1)
            ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0])"""

        # Hyperparameter setup
        degree = kwargs['degree'][ind[0]]
        hyperparams = {}
        hyperparams['degree'] = int(degree)

        # Fit the final model
        X_trans, X_trans_test = rm._feature_trans(X, X_test, degree, kwargs['interaction'], kwargs['trans_type'])
        X_trans = np.hstack((np.ones([X_trans.shape[0], 1]), X_trans))
        X_trans_test = np.hstack((np.ones([X_trans_test.shape[0], 1]), X_trans_test))

        POLY_model, POLY_params, mse_train, mse_test, yhat_train, yhat_test = rm.OLS_fitting(X_trans, y, X_trans_test, y_test)
        return(hyperparams, POLY_model, POLY_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'PLS':
        if not(cv_type.startswith('Group')) and 'K' not in kwargs: # For non-grouped CV types
            kwargs['K'] = np.linspace( 1, min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)), min(X.shape[1], int((K_fold-1)/K_fold * X.shape[0] - 1)) )
        elif 'K' not in kwargs:
            kwargs['K'] = np.linspace(1, min(X.shape[1], X.shape[0]-1), min(X.shape[1], X.shape[0]-1))

        MSE_result = np.zeros((len(kwargs['K']), K_fold*Nr))

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
                PLS = PLSRegression(scale = False, n_components = int(kwargs['K'][i]), tol = eps).fit(X_train_scale, y_train_scale)
                PLS_para = PLS.coef_.reshape(-1,1)
                yhat_val = np.dot(X_val_scale, PLS_para)
                MSE_result[i, counter] = MSE(y_val_scale.flatten(), yhat_val.flatten())

        MSE_mean = np.nanmean(MSE_result, axis = 1)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(MSE_mean), MSE_mean.shape)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 1)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( kwargs['K'] == np.nanmin(kwargs['K'][MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = ind[0][0]

        # Hyperparameter setup
        K = int(kwargs['K'][ind])
        hyperparams = {'K': K}

        # Fit the final model
        PLS_model = PLSRegression(scale = False, n_components = K).fit(X, y)
        PLS_params = PLS_model.coef_.reshape(-1,1)
        yhat_train = np.dot(X, PLS_params)
        yhat_test = np.dot(X_test, PLS_params)
        mse_train = MSE(yhat_train.flatten(), y.flatten())
        mse_test = MSE(yhat_test.flatten(), y_test.flatten())
        return(hyperparams, PLS_model, PLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'ALVEN':
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'label_name' not in kwargs: # Whether to auto-generate label names for the variables [x1, x2, ..., log(x1), ..., 1/x1, ..., x1*x2, etc.]
            kwargs['label_name'] = True # TODO: currently unused. Not sure whether I'll re-implement custom naming
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'all'
        if 'lag' not in kwargs and 'DALVEN' in model_name:
            kwargs['lag'] =  [i+1 for i in range(40)]
        elif model_name == 'ALVEN':
            kwargs['lag'] = [0]

        # First run for variable selection using a L1_ratio of 1 (that is, only using an L1 penalty)
        kwargs['selection'] = None
        MSE_result = np.empty((len(kwargs['degree']) * len(kwargs['alpha']), K_fold*Nr)) * np.nan
        hyperparam_prod = list(product(kwargs['degree'], [1], kwargs['alpha']))
        print(f'Beginning variable selection runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                temp = PAR(delayed(_ALVEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs, counter,
                        prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                MSE_result[:, counter], _ = zip(*temp)
        MSE_mean = np.nanmean(MSE_result, axis = 1)
        ind = np.nanargmin(MSE_mean)
        # Run to obtain the coefficients when ALVEN is run with L1_ratio = 1
        _, ALVEN_params, _, _, _, _, label_names = rm.ALVEN_fitting(X, y, X_test, y_test, hyperparam_prod[ind][2], 1, hyperparam_prod[ind][0], tol = eps,
                                                selection = None, trans_type = kwargs['trans_type'])
        kwargs['selection'] = np.abs(ALVEN_params.flatten()) >= 2.4e-3 # TODO: This value could (should?) depend on the "eps" parameter, but I need to learn more about how it works in other models
        kwargs['degree'] = [hyperparam_prod[ind][0]]

        MSE_result = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan
        Var = np.empty((len(kwargs['alpha']) * len(kwargs['l1_ratio']), K_fold*Nr)) * np.nan # Used when robust_priority == True
        hyperparam_prod = list(product(kwargs['degree'], kwargs['l1_ratio'], kwargs['alpha']))
        print(f'Beginning real runs. There are {len(hyperparam_prod)} hyperparameter combinations')
        with Parallel(n_jobs = -1) as PAR:
            for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                temp = PAR(delayed(_ALVEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, kwargs, counter,
                        prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                MSE_result[:, counter], Var[:, counter] = zip(*temp)

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
        degree = hyperparam_prod[ind][0]
        l1_ratio = hyperparam_prod[ind][1]
        alpha = hyperparam_prod[ind][2]
        hyperparams = {'alpha': alpha, 'l1_ratio': l1_ratio, 'degree': degree}
        # Final run with the test set and best hyperparameters
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, label_names = rm.ALVEN_fitting(X, y, X_test, y_test, alpha,
                                                l1_ratio, degree, tol = eps, selection = kwargs['selection'], trans_type = kwargs['trans_type'])
        # Unscaling the model coefficients as per stackoverflow.com/questions/23642111/how-to-unscale-the-coefficients-from-an-lmer-model-fitted-with-a-scaled-respon
        Xtrans, _, _ = rm._feature_trans(X, degree = degree, trans_type = kwargs['trans_type'])
        ALVEN_params_unscaled = (ALVEN_params.squeeze() * y.std() / Xtrans[:, kwargs['selection']].std(axis=0))
        label_names = label_names[kwargs['selection']]
        # Removing the features that had coefficients equal to zero after the final model selection
        final_selection = np.abs(ALVEN_params.squeeze()) >= 1e-3 # TODO: again, the eps dependency
        ALVEN_params_unscaled = ALVEN_params_unscaled[final_selection].reshape(-1) # Reshape to avoid 0D arrays when only one variable is selected
        label_names = label_names[final_selection].reshape(-1)
        ALVEN_params_unscaled = list(zip(label_names, ALVEN_params_unscaled))
        return(hyperparams, ALVEN_model, ALVEN_params_unscaled, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'RF':
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = [2, 3, 5, 10, 15, 20, 40]
        if 'n_estimators' not in kwargs:
            kwargs['n_estimators'] = [10, 50, 100, 200]
        if 'min_samples_leaf' not in kwargs:
            kwargs['min_samples_leaf'] = [0.0001]#0.02,0.05, 0.1] #, 0.05 ,0.1, 0.2] # 0.3, 0.4]

        MSE_result = np.empty((len(kwargs['max_depth']), len(kwargs['n_estimators']), len(kwargs['min_samples_leaf']), K_fold*Nr))
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the parameters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['max_depth']), len(kwargs['n_estimators']), len(kwargs['min_samples_leaf'])))
            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        S[i, j, k] = i/len(kwargs['max_depth']) - k/len(kwargs['min_samples_leaf'])

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
            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        _, _, mse, _, _ = nro.RF_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, kwargs['n_estimators'][j], kwargs['max_depth'][i], kwargs['min_samples_leaf'][k])
                        MSE_result[i, j, k, counter] = mse

        MSE_mean = np.nanmean(MSE_result, axis = 3)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(MSE_mean), MSE_mean.shape)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 3)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( S == np.nanmin(S[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0], ind[2][0])

        # Hyperparameter setup
        max_depth = kwargs['max_depth'][ind[0]]
        n_estimators = kwargs['n_estimators'][ind[1]]
        min_samples_leaf = kwargs['min_samples_leaf'][ind[2]]
        hyperparams = {}
        hyperparams['max_depth'] = max_depth
        hyperparams['n_estimators'] = n_estimators
        hyperparams['min_samples_leaf'] = min_samples_leaf

        # Fit the final model
        RF_model, mse_train, mse_test, yhat_train, yhat_test = nro.RF_fitting(X, y, X_test, y_test, n_estimators, max_depth, min_samples_leaf)
        return(hyperparams, RF_model, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'SVR':
        if 'C' not in kwargs:
            kwargs['C'] = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        if 'gamma' not in kwargs:
            gd = 1/X.shape[1]
            kwargs['gamma'] = [gd/50, gd/10, gd/5, gd/2, gd, gd*2, gd*5, gd*10, gd*50]
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3]

        MSE_result = np.empty((len(kwargs['C']), len(kwargs['gamma']), len(kwargs['epsilon']), K_fold*Nr)) * np.nan
        if kwargs['robust_priority']:
            # TODO: is this scoring system correct? It ignores the actual values of the parameters, caring only about their lengths and positions in the array.
            S = np.zeros((len(kwargs['C']), len(kwargs['gamma']), len(kwargs['epsilon'])))
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        S[i, j, k] = i/len(kwargs['C']) - j/len(kwargs['gamma']) - k/len(kwargs['epsilon'])

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
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        _, _, mse, _, _ = nro.SVR_fitting(X_train_scale, y_train_scale, X_val_scale, y_val_scale, kwargs['C'][i], kwargs['epsilon'][k], kwargs['gamma'][j])
                        MSE_result[i, j, k, counter] = mse

        MSE_mean = np.nanmean(MSE_result, axis = 3)
        # Min MSE value (first occurrence)
        ind = np.unravel_index(np.nanargmin(MSE_mean), MSE_mean.shape)
        if kwargs['robust_priority']:
            MSE_std = np.nanstd(MSE_result, axis = 3)
            MSE_min = MSE_mean[ind]
            MSE_bar = MSE_min + MSE_std[ind]
            ind = np.nonzero( S == np.nanmin(S[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
            ind = (ind[0][0], ind[1][0], ind[2][0])

        # Hyperparameter setup
        C = kwargs['C'][ind[0]]
        gamma = kwargs['gamma'][ind[1]]
        epsilon = kwargs['epsilon'][ind[2]]
        hyperparams = {}
        hyperparams['C'] = C
        hyperparams['gamma'] = gamma
        hyperparams['epsilon'] = epsilon

        # Fit the final model
        SVR_model, mse_train, mse_test, yhat_train, yhat_test = nro.SVR_fitting(X, y, X_test, y_test, C, epsilon, gamma)
        return(hyperparams, SVR_model, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind])

    elif model_name == 'DALVEN' or model_name == 'DALVEN_full_nonlinear':
        if model_name == 'DALVEN':
            DALVEN = rm.DALVEN_fitting
        else:
            DALVEN = rm.DALVEN_fitting_full_nonlinear
        kwargs['model_name'] = model_name
        if 'degree' not in kwargs:
            kwargs['degree'] = [1, 2, 3]
        if 'lag' not in kwargs:
            kwargs['lag'] =  [i+1 for i in range(40)]
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'all'

        hyperparam_prod = list(product(kwargs['degree'], kwargs['l1_ratio'], range(alpha_num), kwargs['lag']))
        print(f'There are {len(hyperparam_prod)} hyperparameter combinations')

        with Parallel(n_jobs = -1) as PAR:
            if 'IC' in cv_type: # Information criterion
                temp = PAR(delayed(_DALVEN_joblib_fun)(X, y, X_test, y_test, eps, alpha_num, kwargs, prod_idx, this_prod) for prod_idx, this_prod in enumerate(hyperparam_prod))
                temp = list(zip(*temp))[2] # To isolate the (AIC, AICc, BIC) tuple, which is the 3rd subentry of each entry in the original temp
                temp = np.array(temp)
                if cv_type == 'AICc':
                    IC_result = temp[:, 1]
                elif cv_type == 'BIC':
                    IC_result = temp[:, 2]
                else: # AIC
                    IC_result = temp[:, 0]
                # Min IC value (first occurrence)
                ind = np.argmin(IC_result)
            else: # Cross-validation
                MSE_result = np.empty((len(kwargs['degree']) * alpha_num * len(kwargs['l1_ratio']) * len(kwargs['lag']), K_fold*Nr)) * np.nan
                Var = np.empty((len(kwargs['degree']) * alpha_num * len(kwargs['l1_ratio']) * len(kwargs['lag']), K_fold*Nr)) * np.nan
                for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
                    temp = PAR(delayed(_DALVEN_joblib_fun)(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, prod_idx, this_prod, counter) for prod_idx, this_prod in enumerate(hyperparam_prod))
                    MSE_result[:, counter], Var[:, counter], _ = zip(*temp)

                MSE_mean = np.nanmean(MSE_result, axis = 1)
                ind = np.nanargmin(MSE_mean)
                if kwargs['robust_priority']:
                    MSE_std = np.nanstd(MSE_result, axis = 1)
                    MSE_min = MSE_mean[ind]
                    MSE_bar = MSE_min + MSE_std[ind]
                    Var_num = np.nansum(Var, axis = 1)
                    ind = np.nonzero( Var_num == np.nanmin(Var_num[MSE_mean < MSE_bar]) ) # Hyperparams with the lowest number of variables but still within one stdev of the best MSE
                    ind = ind[0][0]
        print('')

        # Hyperparameter setup
        degree = hyperparam_prod[ind][0]
        l1_ratio = hyperparam_prod[ind][1]
        alpha = hyperparam_prod[ind][2]
        lag = hyperparam_prod[ind][3]
        # Final run with the test set and best hyperparameters
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, _ = DALVEN(X, y, X_test, y_test, alpha,
                                                    l1_ratio, degree, lag, alpha_num, tol = eps, cv = False, selection = kwargs['selection'],
                                                    select_value = kwargs['select_value'], trans_type = kwargs['trans_type'])
        hyperparams = {'alpha': alpha, 'l1_ratio': l1_ratio, 'degree': degree, 'lag': lag, 'retain_index': retain_index}
        # Names for the retained variables
        if model_name == 'DALVEN': # DALVEN does transform first, then lag
            Xt, _, label_names = rm._feature_trans(X, degree = degree, trans_type = kwargs['trans_type'])
            # Lag padding for X
            XD = Xt[lag:]
            for i in range(lag):
                XD = np.hstack((XD, Xt[lag-1-i : -i-1]))
            # Lag padding for y in design matrix
            for i in range(lag):
                XD = np.hstack((XD, y[lag-1-i : -i-1]))
        else: # DALVEN_full_nonlinear does lag first, then transform
            # Lag padding for X
            XD = X[lag:]
            for i in range(lag):
                XD = np.hstack((XD, X[lag-1-i : -i-1]))
            # Lag padding for y in design matrix
            for i in range(lag):
                XD = np.hstack((XD, y[lag-1-i : -i-1]))
            XD, _, label_names = rm._feature_trans(XD, degree = degree, trans_type = kwargs['trans_type'])

        # Remove features with insignificant variance
        sel = VarianceThreshold(threshold=eps).fit(XD)
        index = list(sel.get_support())
        label_names = np.matlib.repmat(label_names, 1, lag+1).squeeze()
        for idx in range(1, lag+1):
            label_names[Xt.shape[1]*idx : Xt.shape[1]*(idx+1)] += f'(t-{idx})'
        label_names = np.concatenate(( label_names, [f'y(t-{idx})' for idx in range(1, lag+1)] ))
        label_names = label_names[index][retain_index]

        if 'IC' in cv_type:
            return(hyperparams, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, IC_result[ind], label_names)
        else:
            return(hyperparams, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_mean[ind], label_names)

    elif model_name in {'MLP', 'RNN'}:
        # Loss function
        kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'use_cross_entropy' not in kwargs or not kwargs['use_cross_entropy']:
            loss_function = torch.nn.functional.mse_loss
        else:
            if 'class_weight' not in kwargs or kwargs['class_weight'] is None:
                kwargs['class_weight'] = torch.ones(np.max(y) + int(0 in y)) # If 0 also represents a class, then there are max(y) + 1 classes
            loss_function = torch.nn.CrossEntropyLoss(weight = kwargs['class_weight']).to(kwargs['device'])
        # Layer hyperparameters
        if 'MLP_layers' not in kwargs or kwargs['MLP_layers'] is None:
            myshape_X = X.shape[1]
            if kwargs['use_cross_entropy']:
                myshape_y = np.max(y) + int(0 in y) # TODO: len(set(y)) should work better, especially if one of the classes is missing for some reason
            else:
                myshape_y = y.shape[1]
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
        if 'scheduler_last_epoch' not in kwargs:
            kwargs['scheduler_last_epoch'] = kwargs['n_epochs'] - 30
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
                    print(f'Beginning hyperparameters {cur_idx+1:3}/{len(hyperparam_list)}', end = '\r')
                    if 'scheduler_min_lr' not in kwargs:
                        kwargs['scheduler_min_lr'] = cur_hp[1] / 16
                    temp_val_loss = 0
                    for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X_unscaled, y_unscaled, Type = cv_type, K = K_fold, Nr = Nr, group = group)):
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
                        # Creating the Datasets / DataLoaders
                        train_dataset_fold = MyDataset(X_train_scale, y_train_scale)
                        train_loader_fold = DataLoader(train_dataset_fold, kwargs['batch_size'], shuffle = True)
                        val_dataset_fold = MyDataset(X_val_scale, y_val_scale)
                        val_loader_fold = DataLoader(val_dataset_fold, kwargs['batch_size'], shuffle = True)

                        # Declaring the model and optimizer
                        model = my_ANN(cur_hp[0], cur_hp[2], X_train_scale.shape[-1], kwargs['RNN_layers'], kwargs['device']).to(kwargs['device'])
                        optimizer = torch.optim.AdamW(model.parameters(), lr = cur_hp[1], weight_decay = kwargs['weight_decay'])
                        if kwargs['scheduler'].casefold() == 'plateau':
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, kwargs['scheduler_mode'], kwargs['scheduler_factor'], kwargs['scheduler_patience'], min_lr = kwargs['scheduler_min_lr'])
                        elif kwargs['scheduler'].casefold() == 'cosine':
                            scheduler = CosineScheduler(kwargs['scheduler_last_epoch'], cur_hp[1], warmup_steps = kwargs['scheduler_warmup'], final_lr = kwargs['scheduler_min_lr'])
                        elif kwargs['scheduler'].casefold() == 'lambda':
                            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'step':
                            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, kwargs['scheduler_patience'], kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'multistep':
                            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, kwargs['scheduler_patience'], kwargs['scheduler_factor'])
                        elif kwargs['scheduler'].casefold() == 'exponential':
                            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, kwargs['scheduler_factor'])
                        for epoch in range(kwargs['n_epochs']):
                            train_loss, _ = loop_model(model, optimizer, train_loader_fold, loss_function, epoch, kwargs['batch_size'], categorical = kwargs['use_cross_entropy'])
                            val_loss, _ = loop_model(model, optimizer, val_loader_fold, loss_function, epoch, kwargs['batch_size'], evaluation = True, categorical = kwargs['use_cross_entropy'])
                            if 'scheduler' in locals() and scheduler.__module__ == 'torch.optim.lr_scheduler': # Pytorch built-in scheduler
                                scheduler.step(val_loss)
                            elif 'scheduler' in locals():
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = scheduler(epoch)
                        # Recording the validation loss for this fold
                        temp_val_loss += val_loss
                    # Saving the average validation loss after CV
                    final_val_loss.at[(cur_hp[2], cur_hp[1]), str(cur_hp[0])] = temp_val_loss / (counter+1)
                    # if not nested_validation: # Nested validation requires a bunch of inputs to the same file - I haven't implemented a good way to save the intermediate data
                    final_val_loss.to_csv(kwargs['val_loss_file'])
            return final_val_loss
        final_val_loss = CV_model(X_unscaled, y_unscaled, loss_function, cv_type, K_fold, Nr, group, kwargs)
        # Finding the best hyperparameters
        best_idx, best_LR, best_neurons, best_act, best_act_loc = _get_best_hyperparameters(final_val_loss)
        # Checking whether the best hyperparameters are in the extremes of what was cross-validated
        extreme_LR = best_idx[0] in {best_act_loc.start, best_act_loc.stop-1} # Whether the best LR was either the highest or the lowest value checked for the best activation function
        layer_mask = np.array([len(elem) for elem in kwargs['MLP_layers']]) == len(best_neurons) # Check only MLP configurations with the same number of layers
        layers_for_extreme = np.sort(np.array(kwargs['MLP_layers'], dtype = object)[layer_mask])
        extreme_neuron = best_neurons in [layers_for_extreme[0], layers_for_extreme[-1]]  # Whether the best neuron was either the largest or the smallest combination checked # TODO: MLP configurations with more than 1 hidden layer will benefit from a more thorough analysis of extremity
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
                if len(best_neurons) > 2: # 2 (or more) hidden layers
                    pass
            elif best_neurons == layers_for_extreme[-1]: # The best layer configuration was the largest configuration tested
                new_neurons[0] = (new_neurons[0][0], new_neurons[0][1] + X.shape[1])
                new_neurons[1] = (new_neurons[1][0] + X.shape[1], new_neurons[0][0])
                kwargs['MLP_layers'].append(new_neurons)
                if len(best_neurons) > 2: # 2 (or more) hidden layers
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
        test_dataset = MyDataset(torch.Tensor(X_test), y_test)
        test_loader = DataLoader(test_dataset, kwargs['batch_size'], shuffle = True)
        # Re-declaring the model
        model = my_ANN(best_neurons, best_act, X.shape[-1], kwargs['RNN_layers'], kwargs['device']).to(kwargs['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr = best_LR, weight_decay = kwargs['weight_decay'])
        if kwargs['scheduler'].casefold() in {'plateau', 'cosine'}:
            scheduler = CosineScheduler(kwargs['n_epochs']-30, base_lr = best_LR, warmup_steps = 10, final_lr = best_LR/2)
        # Retrain
        for epoch in range(kwargs['n_epochs']):
            train_loss, train_pred = loop_model(model, optimizer, train_loader, loss_function, epoch, kwargs['batch_size'], categorical = kwargs['use_cross_entropy'])
            if 'scheduler' in locals() and scheduler.__module__ == 'torch.optim.lr_scheduler': # Pytorch built-in scheduler
                scheduler.step(val_loss) # TODO: we do not really have a val_loss here. Need to check how the other built-in Schedulers behave
            elif 'scheduler' in locals():
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)
        # Final evaluation
        test_loss, test_pred = loop_model(model, optimizer, test_loader, loss_function, epoch, kwargs['batch_size'], evaluation = True, categorical = kwargs['use_cross_entropy'])
        return model, final_val_loss, train_loss, test_loss, np.array(train_pred, dtype = float), np.array(test_pred, dtype = float), (str(best_neurons), best_LR, best_act) # Converting to float to save as JSON in SPA.py

        # Old stuff, changes / updates TODO
        """
        if 'IC' in cv_type: # Information criterion
            IC_result = np.zeros( (len(kwargs['activation']), len(kwargs['MLP_layers']), len(kwargs['learning_rate'])) )
            for i in range(len(kwargs['cell_type'])):
                for j in range(len(kwargs['activation'])):
                    for k in range(len(kwargs['RNN_layers'])):
                        _, _, _, (AIC,AICc,BIC), _, _, _ = RNN.timeseries_RNN_feedback_single_train(X, y, X_val, y_val, None, None, kwargs['val_ratio'], kwargs['cell_type'][i],
                                    kwargs['activation'][j], kwargs['RNN_layers'][k], kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'],
                                    kwargs['lambda_l2_reg'], kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test,
                                    output_prob_test, state_prob_test, kwargs['max_checks_without_progress'], kwargs['epoch_before_val'], kwargs['save_location'], plot = False)
                        if cv_type == 'AICc':
                            IC_result[i,j,k] = AICc
                        elif cv_type == 'BIC':
                            IC_result[i,j,k] = BIC
                        else:
                            IC_result[i,j,k] = AIC
            # Min IC value (first occurrence)
            ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)

        # Hyperparameter setup
        cell_type = kwargs['cell_type'][ind[0]]
        activation = kwargs['activation'][ind[1]]
        RNN_layers = kwargs['RNN_layers'][ind[2]]

        prediction_train, prediction_val, prediction_test, _, train_loss_final, val_loss_final, test_loss_final = RNN.timeseries_RNN_feedback_single_train(X, y, None, None, X_test, y_test,
                kwargs['val_ratio'], cell_type, activation, RNN_layers, kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'], kwargs['lambda_l2_reg'],
                kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test, output_prob_test, state_prob_test, kwargs['max_checks_without_progress'],
                kwargs['epoch_before_val'], kwargs['save_location'], kwargs['plot'])

        hyperparams = {}
        hyperparams['cell_type'] = cell_type
        hyperparams['activation'] = activation
        hyperparams['RNN_layers'] = RNN_layers
        hyperparams['training_params'] = {'batch_size': kwargs['batch_size'], 'epoch_overlap': kwargs['epoch_overlap'], 'num_steps': kwargs['num_steps'], 'learning_rate': kwargs['learning_rate'],
                                        'lambda_l2_reg': kwargs['lambda_l2_reg'], 'num_epochs': kwargs['num_epochs']}
        hyperparams['drop_out'] = {'input_prob': kwargs['input_prob'], 'output_prob': kwargs['output_prob'], 'state_prob': kwargs['state_prob']}
        hyperparams['early_stop'] = {'val_ratio': kwargs['val_ratio'], 'max_checks_without_progress': kwargs['max_checks_without_progress'], 'epoch_before_val': kwargs['epoch_before_val']}
        if 'IC' in cv_type:
            hyperparams['IC_optimal'] = IC_result[ind]
        else:
            hyperparams['MSE_val'] = MSE_mean[ind]
        return(hyperparams, kwargs['save_location'], prediction_train, prediction_val, prediction_test, train_loss_final, val_loss_final, test_loss_final)"""

@ignore_warnings()
def _ALVEN_joblib_fun(X_train, y_train, X_val, y_val, eps, kwargs, counter, prod_idx, this_prod):
    """
    A helper function to parallelize ALVEN. Shouldn't be called by the user
    """
    degree, l1_ratio, alpha = this_prod[0], this_prod[1], this_prod[2]
    if prod_idx == 0 or not (prod_idx+1)%200:
        print(f'Beginning run {prod_idx+1:4} of fold {counter+1:3}', end = '\r')
    _, variable, _, mse, _, _, _ = rm.ALVEN_fitting(X_train, y_train, X_val, y_val, alpha, l1_ratio, degree,
                                tol = eps, selection = kwargs['selection'], trans_type = kwargs['trans_type'])
    return mse, np.sum(variable.flatten() != 0)

@ignore_warnings()
def _DALVEN_joblib_fun(X_train, y_train, X_val, y_val, eps, alpha_num, kwargs, prod_idx, this_prod, counter = 0):
    """
    A helper function to parallelize DALVEN. Shouldn't be called by the user
    """
    if prod_idx == 0 or not (prod_idx+1)%200:
        print(f'Beginning run {prod_idx+1:4} of fold {counter+1:3}', end = '\r')
    if kwargs['model_name'] == 'DALVEN':
        _, variable, _, mse, _, _ , _, _, ICs = rm.DALVEN_fitting(X_train, y_train, X_val, y_val, alpha = this_prod[2], l1_ratio = this_prod[1],
                                    degree = this_prod[0], lag = this_prod[3], tol = eps , alpha_num = alpha_num, cv = True, selection = kwargs['selection'],
                                    select_value = kwargs['select_value'], trans_type = kwargs['trans_type'], use_cross_entropy = kwargs['use_cross_entropy'])
    else:
        _, variable, _, mse, _, _ , _, _, ICs = rm.DALVEN_fitting_full_nonlinear(X_train, y_train, X_val, y_val, alpha = this_prod[2], l1_ratio = this_prod[1],
                                    degree = this_prod[0], lag = this_prod[3], tol = eps , alpha_num = alpha_num, cv = True, selection = kwargs['selection'],
                                    select_value = kwargs['select_value'], trans_type = kwargs['trans_type'], use_cross_entropy = kwargs['use_cross_entropy'])
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
def loop_model(model, optimizer, loader, loss_function, epoch, batch_size, evaluation = False, categorical = False):
    if evaluation:
        model.eval()
    else:
        model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_loss = 0
    for idx, data in enumerate(loader):
        X, y = data
        X = X.to(device)
        y = y.to(device)
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
        loss = loss_function(pred, y)
        # Backpropagation
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += y.shape[0]*loss.item()/loader.dataset.ydata.shape[0]
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
            self.lstm = [torch.nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True, bidirectional=True).to(device)] # Need to send to device because the cells are in a list
        elif isinstance(lstm_hidden_size, (list, tuple)):
            self.lstm = []
            for idx, size in enumerate(lstm_hidden_size):
                if idx == 0:
                    self.lstm.append(torch.nn.LSTM(lstm_input_size, size, batch_first=True, bidirectional=True).to(device)) # Need to send to device because the cells are in a list
                else:
                    self.lstm.append(torch.nn.LSTM(lstm_hidden_size[idx-1], size, batch_first=True, bidirectional=True).to(device)) # Need to send to device because the cells are in a list
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
