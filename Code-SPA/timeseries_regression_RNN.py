"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import RNN_feedback as RNN_fd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import tensorflow as tf

# Seed value
# TODO: do we really need to set up all these seeds?
seed_value = 1
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
seed_value += 1
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
seed_value += 1
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
seed_value += 1
# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)


def timeseries_RNN_feedback_single_train(X_train, y_train, X_val = None, y_val = None, X_test = None, y_test = None, val_ratio = 0.2, cell_type = 'basic', activation = 'tanh',
                                            RNN_layers = [512, 256], batch_size = 1, epoch_overlap = None, num_steps = 10, learning_rate = 1e-3, lambda_l2_reg = 1e-3,
                                            num_epochs = 200, input_prob = 0.95, output_prob = 0.95, state_prob = 0.95, input_prob_test = 1, output_prob_test = 1,
                                            state_prob_test = 1, max_checks_without_progress = 100, epoch_before_val = 50, save_location = 'RNN_feedback_0', plot = False):
    """
    Fits an RNN_feedback model to training data, using validation data for early stopping.
    Test data, if given, are used to choose the hyperparameters.
    Otherwise, AIC is returned based on the training data to select the hyperparameters.
    
    Parameters
    ----------
    X_train, y_train : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_val, y_val : Numpy array with shape N_val x m, N_val x 1, optional, default = None
        Validation data predictors and response.
        If None, X_val and y_val are generated from the training data based on val_ratio.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1, optional, default = None
        Testing data predictors and response.
        If None, the RNN is not tested.
    val_ratio: float, optional, default = 0.2
        The fraction of training data used to validate the model.
        The rest are used as training data.
        Relevant only when X_val is None
    cell_type : str, optional, default = 'basic'
        RNN cell used. Should be LSTM, GRU, or basic.
    activation: str, optional, default = 'tanh'
        NN activation function. Should be relu, tanh, sigmoid, or linear.
    RNN_layers : array, optional, default = [512, 256]
        An array with the number of neurons in each layer.
        The length of this array is the number of hidden layers.
    batch_size : int, optional, default = 1
        Batch size used when training the RNN.
    epoch_overlap : None or int, optional, default = None
        If None, there will be no overlap between each training patch.
        If int, there will be epoch_overlap spaces between each path, (e.g.: 0 represents adjacent patch)
    num_steps : int, optional, default = 10
        Number of steps in memory when training a dynamic RNN
    learning_rate : float, optional, default = 1e-3
        Learning rate for the Adam optimizer
    lambda_l2_reg : float, optional, default = 1e-3
        Regularization weight. 0 indicates no regularization.
    num_epochs : int, optional, default = 200
        Maximum number of epochs.
    input_prob, output_prob, state_prob : floats between 0 and 1, optional, default = 0.95
        The keep probability for dropout during training.
    input_prob_test, output_prob_test, state_prob_test : floats between 0 and 1, optional, default = 1
        The keep probability for dropout during testing.
    max_chekcs_without_progress: int, optional, default = 100
        Maximum number of epochs without validation error improvement before early stopping.
    epoch_before_val: int, optional, default = 50
        Minimum number of epochs before using validation set to early stop.
    save_location: str, optional, default = 'RNN_feedback_0'
        Path where the trained RNN will be saved.
    plot: bool, optional, default = False
        Whether to plot the training results.
    """

    # Load and pre-process the data
    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_ratio) 
    num_train = X_train.shape[0]
    if X_test is not None:
        num_test = X_test.shape[0]
    x_num_features = X_train.shape[1]
    y_num_features = y_train.shape[1]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val) 
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_train = scalery.transform(y_train)
    y_val = scalery.transform(y_val)
    if X_test:
        X_test = scaler.transform(X_test)
        y_test = scalery.transform(y_test)

    g_train = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, num_steps, x_num_features, y_num_features, learning_rate, lambda_l2_reg)
    train_loss, val_loss, num_parameter = RNN_fd.train_rnn(X_train, y_train, X_val, y_val, g_train, num_epochs, num_steps, batch_size, input_prob, output_prob, state_prob, 
                                                      epoch_before_val, max_checks_without_progress, epoch_overlap, verbose = True, save = save_location)
    val_loss = np.array(val_loss)
    if plot:
        plt.figure()
        s = 12
        plt.plot(train_loss, color = 'xkcd:sky blue', label = 'train loss')
        plt.plot(np.linspace(epoch_before_val-1, epoch_before_val+val_loss.shape[0]-1, num = val_loss.shape[0]), val_loss, color = 'xkcd:coral', label = 'val loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch Number')
        plt.legend(fontsize = s)
        plt.tight_layout()
        plt.savefig('Training_and_validation_error_round1.png', dpi = 600, bbox_inches = 'tight')
                 
    # Training: final results
    g_train_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, num_train, x_num_features, y_num_features, learning_rate, lambda_l2_reg)
    prediction_train, train_loss_final, _ = RNN_fd.test_rnn(X_train, y_train, g_train_final, save_location, input_prob_test, output_prob_test, state_prob_test, num_train)
    
    AIC = num_train*np.log(np.sum(train_loss_final)/y_num_features) + 2*num_parameter
    AICc = num_train*np.log(np.sum(train_loss_final)/y_num_features) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train)
    BIC = num_train*np.log(np.sum(train_loss_final)/y_num_features) + num_parameter*np.log(num_train)

    # Validation: final results
    g_val_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, X_val.shape[0], x_num_features, y_num_features, learning_rate, lambda_l2_reg)
    prediction_val, val_loss_final, _ = RNN_fd.test_rnn(X_val, y_val, g_val_final, save_location, input_prob_test, output_prob_test, state_prob_test, X_val.shape[0])

    # Testing results
    if X_test:
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, X_test.shape[0], x_num_features, y_num_features, learning_rate, lambda_l2_reg)
        prediction_test, test_loss_final, _ = RNN_fd.test_rnn(X_test, y_test, g_test, save_location, input_prob_test, output_prob_test, state_prob_test, X_test.shape[0])
    else:
        prediction_test = None
        test_loss_final = None    

    # Plotting the results 
    if plot:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('Paired')
        plot_types = {'training', 'validation'}
        if X_test:
            plot_types.add('testing')
        # Prediction vs real
        for j in range(y_num_features):
            for ptype in plot_types:
                if ptype == 'training':
                    y_var = y_train[1:, j]
                    pred_var = prediction_train[1:, j]
                elif ptype == 'validation':
                    y_var = y_val[1:, j]
                    pred_var = prediction_val[1:, j]
                elif ptype == 'testing':
                    y_var = y_test[1:, j]
                    pred_var = prediction_test[1:, j]
                plot_helper(y_var, pred_var, cmap, j, ptype, s)
                
    return (prediction_train, prediction_val, prediction_test, (AIC,AICc,BIC), train_loss_final, val_loss_final, test_loss_final)

def timeseries_RNN_feedback_multi_train(X_train, y_train, timeindex_train, X_val = None, y_val = None, timeindex_val = None, X_test = None, y_test = None, val_ratio = 0.2,
                                            cell_type = 'basic', activation = 'tanh', RNN_layers = [512, 256], batch_size = 1, epoch_overlap = None, num_steps = 10,
                                            learning_rate = 1e-3, lambda_l2_reg = 1e-3, num_epochs = 200, input_prob = 0.95, output_prob = 0.95, state_prob = 0.95,
                                            input_prob_test = 1, output_prob_test = 1, state_prob_test = 1, max_checks_without_progress = 100, epoch_before_val = 50,
                                            save_location = 'RNN_feedback_0', plot = False):
    """
    Fits an RNN_feedback model to training data, using validation data for early stopping.
    Test data, if given, are used to choose the hyperparameters.
    Otherwise, AIC is returned based on the training data to select the hyperparameters.
    
    Parameters
    ----------
    X_train, y_train : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    timeindex_train: dictionary
        Starting from 1, each entry is the time index for that series.
    X_val, y_val : Numpy array with shape N_val x m, N_val x 1, optional, default = None
        Validation data predictors and response.
        If None, X_val and y_val are generated from the training data based on val_ratio.
    timeindex_val: dictionary
        Starting from 1, each entry is the time index for that series.
        If None, timeindex_val is set to timeindex_train
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1, optional, default = None
        Testing data predictors and response.
        If None, the RNN is not tested.
    val_ratio: float, optional, default = 0.2
        The fraction of training data used to validate the model.
        The rest are used as training data.
        Relevant only when X_val is None
    cell_type : str, optional, default = 'basic'
        RNN cell used. Should be LSTM, GRU, or basic.
    activation: str, optional, default = 'tanh'
        NN activation function. Should be relu, tanh, sigmoid, or linear.
    RNN_layers : array, optional, default = [512, 256]
        An array with the number of neurons in each layer.
        The length of this array is the number of hidden layers.
    batch_size : int, optional, default = 1
        Batch size used when training the RNN.
    epoch_overlap : None or int, optional, default = None
        If None, there will be no overlap between each training patch.
        If int, there will be epoch_overlap spaces between each path, (e.g.: 0 represents adjacent patch)
    num_steps : int, optional, default = 10
        Number of steps in memory when training a dynamic RNN
    learning_rate : float, optional, default = 1e-3
        Learning rate for the Adam optimizer
    lambda_l2_reg : float, optional, default = 1e-3
        Regularization weight. 0 indicates no regularization.
    num_epochs : int, optional, default = 200
        Maximum number of epochs.
    input_prob, output_prob, state_prob : floats between 0 and 1, optional, default = 0.95
        The keep probability for dropout during training.
    input_prob_test, output_prob_test, state_prob_test : floats between 0 and 1, optional, default = 1
        The keep probability for dropout during testing.
    max_chekcs_without_progress: int, optional, default = 100
        Maximum number of epochs without validation error improvement before early stopping.
    epoch_before_val: int, optional, default = 50
        Minimum number of epochs before using validation set to early stop.
    save_location: str, optional, default = 'RNN_feedback_0'
        Path where the trained RNN will be saved.
    plot: bool, optional, default = False
        Whether to plot the training results.
    """
    # Load and pre-process the data
    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_ratio) 
    if timeindex_val is None:
        timeindex_val = timeindex_train
    num_train = X_train.shape[0]
    if X_test is not None:
        num_test = X_test.shape[0]
    x_num_features = X_train.shape[1]
    y_num_features = y_train.shape[1]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val) 
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_train = scalery.transform(y_train)
    y_val = scalery.transform(y_val)
    if X_test:
        X_test = scaler.transform(X_test)
        y_test = scalery.transform(y_test)
    if plot:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('Paired')
    
    g_train = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, num_steps, x_num_features, y_num_features, learning_rate, lambda_l2_reg)
    train_loss, val_loss, num_parameter = RNN_fd.train_rnn_multi(X_train, y_train, X_val, y_val, timeindex_train, timeindex_val, g_train, num_epochs, num_steps, batch_size, input_prob,
                                                                output_prob, state_prob, epoch_before_val, max_checks_without_progress, epoch_overlap, verbose = True, save = save_location)
    val_loss = np.array(val_loss)
    if plot:
        plt.figure()
        s = 12
        plt.plot(train_loss, color = 'xkcd:sky blue', label = 'train loss')
        plt.plot(np.linspace(epoch_before_val-1, epoch_before_val+val_loss.shape[0]-1, num = val_loss.shape[0]), val_loss, color = 'xkcd:coral', label = 'val loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch Number')
        plt.legend(fontsize = s)
        plt.tight_layout()
        plt.savefig('Training_and_validation_error_round1.png', dpi = 600, bbox_inches = 'tight')

    # Training: final results
    cum = 0
    prediction_train = []
    train_loss = []
    train_loss_final = []
    for index in range(len(timeindex_train)):
        num = np.shape(timeindex_train[index+1])[0] 
        g_train_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, num, x_num_features, y_num_features, learning_rate, lambda_l2_reg)
        temp_train_pred, temp_train_loss, _ = RNN_fd.test_rnn(X_train[cum:cum+num], y_train[cum:cum+num], g_train_final, save_location, input_prob_test, output_prob_test, state_prob_test, num)

        prediction_train.append(temp_train_pred)
        train_loss.append(temp_train_loss*num)
        train_loss_final.append(temp_train_loss)
        
        if plot:
            for j in range(y_num_features):
                y_var = y_train[cum+1 : cum+num, j]
                pred_var = temp_train_pred[1:, j]
                plot_helper(y_var, pred_var, cmap, j, 'training', s)
        cum += num
        
    AIC = cum*np.log(np.sum(train_loss)/cum/y_num_features) + 2*num_parameter
    AICc = cum*np.log(np.sum(train_loss)/cum/y_num_features) + (num_parameter+cum)/(1-(num_parameter+2)/cum)
    BIC = cum*np.log(np.sum(train_loss)/cum/y_num_features) + np.log(cum)*num_parameter

    # Validation: final results
    cum = 0
    prediction_val = []
    val_loss_final = []
    for index in range(len(timeindex_val)):
        num = np.shape(timeindex_val[index+1])[0] 
        g_val_final = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, num, x_num_features, y_num_features, learning_rate, lambda_l2_reg)
        temp_val_pred, temp_val_loss, _ = RNN_fd.test_rnn(X_val[cum:cum+num], y_val[cum:cum+num], g_val_final, save_location, input_prob_test, output_prob_test, state_prob_test, num)

        prediction_val.append(temp_val_pred)
        val_loss_final.append(temp_val_loss)
        
        if plot:
            for j in range(y_num_features):
                y_var = y_val[cum+1 : cum+num, j]
                pred_var = temp_val_pred[1:, j]
                plot_helper(y_var, pred_var, cmap, j, 'validation', s)
        cum += num

    # Testing results
    if X_test:
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, X_test.shape[0], x_num_features, y_num_features, learning_rate, lambda_l2_reg)
        prediction_test, test_loss_final, _ = RNN_fd.test_rnn(X_test, y_test, g_test, save_location, input_prob_test, output_prob_test, state_prob_test, X_test.shape[0])
        if plot:
            for j in range(y_num_features):
                y_var = y_test[1:, j]
                pred_var = prediction_test[1:, j]
                plot_helper(y_var, pred_var, cmap, j, 'testing', s)
    else:
        prediction_test = None
        test_loss_final = None    
    
    return (prediction_train, prediction_val, prediction_test, (AIC,AICc,BIC), train_loss_final, val_loss_final, test_loss_final)

def timeseries_RNN_feedback_test(X_train, y_train, X_test, y_test, kstep = 1, cell_type = 'basic', activation = 'tanh', RNN_layers = [512, 256],
                                        input_prob_test = 1, output_prob_test = 1, state_prob_test = 1, save_location = 'RNN_feedback_0', plot = False):
    """
    TODO: documentation   
 
    Parameters
    ----------
    X_train, y_train : Numpy array with shape N x m, N x 1
        Training data predictors and response. Used only to scale the test data.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    kstep : int, optional, default = 1
        Number of steps ahead for prediction.
        The output at time t is calculated using previous outputs ...
        up to time t-K and inputs up to time t.
    cell_type : str, optional, default = 'basic'
        RNN cell used. Should be LSTM, GRU, or basic.
    activation: str, optional, default = 'tanh'
        NN activation function. Should be relu, tanh, sigmoid, or linear.
    RNN_layers : array, optional, default = [512, 256]
        An array with the number of neurons in each layer.
        The length of this array is the number of hidden layers.
    input_prob_test, output_prob_test, state_prob_test : floats between 0 and 1, optional, default = 1
        The keep probability for dropout during testing.
    save_location: str, optional, default = 'RNN_feedback_0'
        Path where the trained RNN will be saved.
    plot: bool, optional, default = False
        Whether to plot the training results.
    """
    # Load and pre-process the data
    x_num_features = X_train.shape[1]
    y_num_features = y_train.shape[1]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    scalery = StandardScaler()
    scalery.fit(y_train)
    y_test = scalery.transform(y_test)
    
    kstep -= 1 # Adjustment for the test_rnn code to be comparable with MATLAB

    # Testing with k = 0
    if len(RNN_layers) == 1:
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, X_test.shape[0], x_num_features, y_num_features, learning_rate = 0, lambda_l2_reg = 0)
        test_y_prediction, test_loss_final, test_rnn_outputs = RNN_fd.test_rnn(X_test, y_test, g_test, save_location, input_prob_test, output_prob_test, state_prob_test, X_test.shape[0])
    else:
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, 1, x_num_features, y_num_features, learning_rate = 0, lambda_l2_reg = 0) # TODO: is num_steps really 1 here?
        test_y_prediction, test_loss_final, test_rnn_outputs = RNN_fd.test_rnn_layer(X_test, y_test, g_test, save_location, input_prob_test, output_prob_test, state_prob_test, len(RNN_layers)) # rnn_outputs = inter_state

    # Testing with k = k
    if kstep > 0: 
        g_test = RNN_fd.build_multilayer_rnn_graph_with_dynamic_rnn(cell_type, RNN_layers, activation, 1, x_num_features, y_num_features, learning_rate = 0, lambda_l2_reg = 0)
        test_y_prediction_kstep, test_loss_kstep = RNN_fd.test_rnn_kstep(X_test, y_test, test_y_prediction, test_rnn_outputs, g_test, save_location, input_prob_test,
                                                                        output_prob_test, state_prob_test, X_test.shape[0], kstep)
    else:
        test_y_prediction_kstep = None
        test_loss_kstep = None
    
    loss_final = np.vstack((test_loss_final,test_loss_kstep))
    prediction_final = {}
    predition_final[1] = test_y_prediction
    for i in range(1, kstep+1):
        prediction_final[i+1] = test_y_prediction_kstep[i]
    
    if plot: 
        from matplotlib.cm import get_cmap
        cmap = get_cmap('Paired')
        s = 12
        
        if X.shape[0] == X_test.shape[0] and np.sum(X - X_test) < 1e-4:
            name = 'Train' +round_number
        else: 
            name = 'Test' + round_number
            
        #plot the prediction vs real
        for i in range(kstep+1):
            for j in range(y_num_features):
                y_var = y_test[i+1:, j]
                pred_var = prediction_final[i][1:, j] 
                plot_helper(y_var, pred_var, cmap, j, name, s, i)

        #plot fitting errors
        max_limit = np.max(prediction_final[kstep][kstep+1:], axis = 0)
        min_limit = np.min(prediction_final[kstep][kstep+1:], axis = 0)
        fig2, axs2 = plt.subplots(kstep+1, y_num_features, figsize = (3*y_num_features, 2*(kstep+1)))
        
        if y_num_features > 1:
            for i in range(kstep+1):
                for j in range(y_num_features):
                    axs2[i,j].plot(prediction_final[i][1:, j] - Y_test[i+1:, j], color = cmap(j*2+1))
                    axs2[i,j].set_title(f'{name} data step{i+1} error for y{j+1}', fontsize = s)
                    axs2[i,j].set_ylim(min_limit[j] - abs(min_limit[j])*0.5, max_limit[j]*1.5)
                    if i is kstep-1:
                        axs2[i,j].set_xlabel('Time index', fontsize = s)
        else: # TODO: can we merge these two for loops, or will an error appear because the variables below aren't 2D?
            for i in range(kstep+1):
                axs2[i].plot(prediction_final[i][1:]-Y_test[i+1:], color= cmap(2+1))
                axs2[i].set_title(f'{name} data step{i+1} error for y1', fontsize = s)
                axs2[i].set_ylim(min_limit-abs(min_limit)*0.5,max_limit*1.5)
                if i is kstep-1:
                    axs2[i].set_xlabel('Time index', fontsize=s)                
        fig2.tight_layout()
        plt.savefig(f'{name}_error_kstep.png', dpi = 600, bbox_inches = 'tight')        
        
        # MSE for prediction results over different steps
        MSE_test = np.vstack((test_loss_final, test_loss_kstep))
        for i in range(y_num_features):
            plt.figure(figsize = (3,2))
            plt.plot(range(1, MSE_test.shape[0]), MSE_test[:,i], 'd-', color = cmap(i*2+1))
            plt.title(f'{name} MSE for y{i} prediction', fontsize = s)
            plt.xlabel('k-steps ahead', fontsize = s)
            plt.ylabel('MSE', fontsize = s)
            plt.tight_layout()                    
            plt.savefig(f'MSE_{name}_var_{i}.png', dpi = 600, bbox_inches = 'tight')
    
    return (prediction_final, loss_final)

def plot_helper(y_var, pred_var, cmap, j, ptype, s = 12, i = -1):
    plt.figure(figsize = (5,3))
    plt.plot(y_var, color = cmap(j*2+1), label = 'real')
    plt.plot(pred_var, '--', color = 'xkcd:coral', label = 'prediction')
    if i < 0: # Proxy for not(called by the single_train function) 
        plt.title(f'RNN {ptype} data prediction for y{j+1}', fontsize = s)
    else:
        plt.title(f'{ptype} data step{i+1} prediction for y{j+1}', fontsize = s)
    plt.xlabel('Time index', fontsize = s)
    plt.ylabel('y', fontsize = s)
    plt.legend(fontsize = s)
    plt.tight_layout()
    plt.savefig(f'RNN_{ptype}_var_{j+1}.png', dpi = 600, bbox_inches = 'tight')

    if i < 0:
        plt.figure(figsize = (5,3))
        plt.plot(pred_var - y_var, color = cmap(j*2+1))
        plt.title(f'RNN {ptype} error for y{j+1}', fontsize = s)
        plt.xlabel('Time index', fontsize = s)
        plt.ylabel('prediction - real', fontsize = s)
        plt.tight_layout()
        plt.savefig(f'RNN_{ptype}_var_{j+1}_error.png', dpi = 600, bbox_inches = 'tight')

