"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
from sklearn.feature_selection import VarianceThreshold
import regression_models as rm
import numpy as np
import nonlinear_regression as nr
import timeseries_regression_RNN as RNN

def IC_mse(model_name, X, y, X_test, y_test, X_val = None, y_val = None, cv_type = None, alpha_num = 50, eps = 1e-4, **kwargs):
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
    X_val, y_val : Numpy array with shape N_val x m, N_val x 1, optional, default = None
        Validation data predictors and response.
        If None and model_name == 'RNN', X_val and y_val are generated from the training data based on val_ratio.
    cv_type : str, optional, default = None
        The information criterion used for cross-validation.
        One of {'BIC', 'AIC', 'AICc'}. If None, 'AIC' or 'AICc' is used based on X data size.
    alpha_num : int, optional, default = 50
        Penalty weight used when model_name in {'RR', 'EN', 'ALVEN', 'DALVEN'}.
    eps : float, optional, default = 1e-4
        Tolerance. TODO: expand on this.
    **kwargs : dict, optional
        Non-default hyperparameters for model fitting.
    """    
    if cv_type is None:
        if X.shape[0]//X.shape[1] < 40:
            cv_type = 'AICc'
        else:
            cv_type = 'AIC'

    if model_name == 'DALVEN' or model_name == 'DALVEN_full_nonlinear':
        DALVEN = rm.model_getter(model_name)
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2,3]
        if 'lag' not in kwargs:
            kwargs['lag'] =  [i+1 for i in range(40)]
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
 
        IC_result = np.zeros( (len(kwargs['degree']), alpha_num, len(kwargs['l1_ratio']), len(kwargs['lag'])) )
        for k in range(len(kwargs['degree'])):
            for j in range(len(kwargs['l1_ratio'])):
                for i in range(alpha_num):
                    for t in range(len(kwargs['lag'])):
                        _, _, _, _, _, _ , _, _, (AIC,AICc,BIC) = DALVEN(X, y, X_test, y_test, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                        if cv_type == 'AICc': # TODO: change to just = instead of += ?
                            IC_result[k,i,j,t] += AICc
                        elif cv_type == 'BIC':
                            IC_result[k,i,j,t] += BIC
                        else:
                            IC_result[k,i,j,t] += AIC

        # Min IC value (first occurrence)
        ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)
        degree = kwargs['degree'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[2]]
        lag = kwargs['lag'][ind[3]]
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, _ = DALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree = degree, lag = lag, tol = eps, alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
        hyperparams = {}
        hyperparams['alpha'] = alpha
        hyperparams['l1_ratio'] = l1_ratio
        hyperparams['degree'] = degree
        hyperparams['lag'] = lag
        hyperparams['retain_index'] = retain_index

        # Names for the retained variables(?)
        if kwargs['label_name'] :
            if model_name == 'DALVEN': # TODO: is there really a difference between those two? Should there be a difference?
                if kwargs['trans_type'] == 'auto':
                    Xtrans, _ = nr.feature_trans(X, degree = degree, interaction = 'later')
                else:
                    Xtrans, _ = nr.poly_feature(X, degree = degree, interaction = True, power = True)
                # Lag padding for X
                XD = Xtrans[lag:]
                for i in range(lag):
                    XD = np.hstack((XD, Xtrans[lag-1-i:-i-1]))
            else: # DALVEN_full_nonlinear
                # Lag padding for X
                XD = X[lag:]
                for i in range(lag):
                    XD = np.hstack((XD, X[lag-1-i:-i-1]))

            # Lag padding for y in design matrix
            for i in range(lag):
                XD = np.hstack((XD, y[lag-1-i:-i-1]))

            if model_name == 'DALVEN_full_nonlinear': # TODO: again, is there really a difference between those two? Should there be a difference?
                if kwargs['trans_type'] == 'auto':
                    XD,_ = nr.feature_trans(XD, degree = degree, interaction = 'later')
                else:
                    XD, _ = nr.poly_feature(XD, degree = degree, interaction = True, power = True)

            # Remove features with insignificant variance
            sel = VarianceThreshold(threshold=eps).fit(XD)

            if 'xticks' in kwargs:
                list_name = kwargs['xticks']
            else:
                list_name = [f'x{i}' for i in range(1, np.shape(X)[1]+1)]
            
            list_name_final = list_name[:] # [:] makes a copy
            if kwargs['trans_type'] == 'auto':
                list_name_final += [f'log({name})' for name in list_name] + [f'sqrt({name})' for name in list_name] + [f'1/{name}' for name in list_name]

                if degree >= 2:
                    for i in range(X.shape[1]-1):
                        for j in range(i+1, X.shape[1]):
                            list_name_final += [f'{list_name[i]}*{list_name[j]}']
                    list_name_final += [f'{name}^2' for name in list_name] + [f'(log{name})^2' for name in list_name] + [f'1/{name}^2' for name in list_name] + (
                            [f'{name}^1.5' for name in list_name] + [f'log({name})/{name}' for name in list_name]+ [f'1/{name}^0.5' for name in list_name] )

                if degree >= 3:
                    for i in range(X.shape[1]-2):
                        for j in range(i+1, X.shape[1]-1):
                            for k in range(j+1, X.shape[1]):
                                list_name_final += [f'{list_name[i]}*{list_name[j]}*{list_name[k]}']
                    list_name_final += [f'{name}^3' for name in list_name] + [f'(log{name})^3' for name in list_name] + [f'1/{name}^3' for name in list_name] + (
                                [f'{name}^2.5' for name in list_name] + [f'(log{name})^2/{name}' for name in list_name]+ [f'log({name})/sqrt({name})' for name in list_name] +
                                [f'log({name})/{name}^2' for name in list_name] + [f'{name}^-1.5' for name in list_name]
                        
            elif degree >= 2:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        list_name_final += [f'{list_name[i]}*{list_name[j]}']
                if degree >= 3:
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            for k in range(j, X.shape[1]):
                                list_name_final += [f'{list_name[i]}*{list_name[j]}*{list_name[k]}']
                        
            list_copy = list_name_final[:]
            for i in range(lag):
                list_name_final += [f'{s}(t-{i+1})' for s in list_copy]
            for i in range(lag):
                list_name_final += [f'y(t-{i+1})'] 
                                
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final = [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []
        return(hyperparams, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, IC_result[ind], list_name_final)

    # For RNN, only the model archetecture is viewed as hyper-parameter in thie automated version, the other training parameters can be set by kwargs, otw the default value will be used
    elif model_name == 'RNN':
        input_size_x = X.shape[1]
        
        # Model architecture
        if 'cell_type' not in kwargs:
            kwargs['cell_type'] = ['basic']
        if 'activation' not in kwargs:
            kwargs['activation'] = ['tanh']
        if 'RNN_layers' not in kwargs:
            kwargs['RNN_layers'] = [input_size_x]

        # Training parameters
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 1
        if 'epoch_overlap' not in kwargs:
            kwargs['epoch_overlap'] = None
        if 'num_steps' not in kwargs:
            kwargs['num_steps'] = 10
        if 'learning_rate' not in kwargs:
            kwargs['learning_rate'] = 1e-3
        if 'lambda_l2_reg' not in kwargs:
            kwargs['lambda_l2_reg'] = 1e-3
        if 'num_epochs' not in kwargs:
            kwargs['num_epochs'] = 200
        # Dropout parameters
        if 'input_prob' not in kwargs:
            kwargs['input_prob'] = 0.95
        if 'output_prob' not in kwargs:
            kwargs['output_prob'] = 0.95
        if 'state_prob' not in kwargs:
            kwargs['state_prob'] = 0.95
        # Currently we do not support BRNNs, so always keep all neurons during test
        input_prob_test = 1
        output_prob_test = 1
        state_prob_test = 1

        # Early stopping
        if 'val_ratio' not in kwargs and X_val is None:
            kwargs['val_ratio'] = 0.2
        else:
            kwards['val_ratio'] = 0
        if 'max_checks_without_progress' not in kwargs:
            kwargs['max_checks_without_progress'] = 50
        if 'epoch_before_val' not in kwargs:
            kwargs['epoch_before_val'] = 50

        if 'save_location' not in kwargs:
            kwargs['save_location'] = 'RNN_feedback_0'
        if 'plot' not in kwargs:
            kwargs['plot'] = False
        
        IC_result = np.zeros( (len(kwargs['cell_type']), len(kwargs['activation']), len(kwargs['RNN_layers'])) )
        Result = {}
        
        for i in range(len(kwargs['cell_type'])):
            for j in range(len(kwargs['activation'])):
                for k in range(len(kwargs['RNN_layers'])):
                    p_train, p_val, p_test, (AIC,AICc,BIC), train_loss, val_loss, test_loss = RNN.timeseries_RNN_feedback_single_train(X, y, X_val, y_val, X_test, y_test, kwargs['val_ratio'],
                            kwargs['cell_type'][i], kwargs['activation'][j], kwargs['RNN_layers'][k], kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'],
                            kwargs['lambda_l2_reg'], kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test, output_prob_test, state_prob_test,
                            kwargs['max_checks_without_progress'], kwargs['epoch_before_val'], kwargs['location'], plot = False)
                        if cv_type == 'AICc': # TODO: change to just = instead of += ?
                            IC_result[i,j,k] += AICc
                        elif cv_type == 'BIC':
                            IC_result[i,j,k] += BIC
                        else:
                            IC_result[i,j,k] += AIC
                            
                        Result[(i,j,k)] = {'prediction_train': p_train, 'prediction_val': p_val, 'prediction_test': p_test,
                                            'train_loss_final': train_loss, 'val_loss_final': val_loss, 'test_loss_final': test_loss}
        
        # Min IC value (first occurrence)
        ind = np.unravel_index(np.argmin(IC_result, axis=None), IC_result.shape)
        cell_type = kwargs['cell_type'][ind[0]]
        activation = kwargs['activation'][ind[1]]
        RNN_layers = kwargs['RNN_layers'][ind[2]]
       
        Final = Result[(ind[0],ind[1],ind[2])]
        prediction_train, prediction_val, prediction_test, AICs, train_loss_final, val_loss_final, test_loss_final = RNN.timeseries_RNN_feedback_single_train(X, y, X_val, y_val, X_test, y_test,
                kwargs['val_ratio'], cell_type, activation, RNN_layers, kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'], kwargs['lambda_l2_reg'],
                kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test, output_prob_test, state_prob_test, kwargs['max_checks_without_progress'],
                kwargs['epoch_before_val'], kwargs['location'], plot = False)
        
        hyperparams = {}
        hyperparams['cell_type'] = cell_type
        hyperparams['activation'] = activation
        hyperparams['RNN_layers'] = RNN_layers
        hyperparams['training_params'] = {'batch_size': kwargs['batch_size'], 'epoch_overlap': kwargs['epoch_overlap'], 'num_steps': kwargs['num_steps'], 'learning_rate': kwargs['learning_rate'],
                                        'lambda_l2_reg': kwargs['lambda_l2_reg'], 'num_epochs': kwargs['num_epochs']}
        hyperparams['drop_out'] = {'input_prob': kwargs['input_prob'], 'output_prob': kwargs['output_prob'], 'state_prob': kwargs['state_prob']}
        hyperparams['early_stop'] = {'train_ratio': kwargs['train_ratio'], 'max_checks_without_progress': kwargs['max_checks_without_progress'], 'epoch_before_val': kwargs['epoch_before_val']}
        hyperparams['IC_optimal'] = IC_result[ind]
        return(hyperparams, kwargs['location'], Final['prediction_train'], Final['prediction_val'], Final['prediction_test'], Final['train_loss_final'], Final['val_loss_final'], Final['test_loss_final'])

