"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import statsmodels.api as sm
from SPLS_Python import SPLS
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import numpy.matlib as matlib
import nonlinear_regression as nr
import warnings
warnings.filterwarnings("ignore") # TODO: Want to just ignore the PLS constant residual warnings, but this will do for now

def model_getter(model_name):
    '''Return the model according to the name'''
    switcher = {
            'OLS': OLS_fitting,
            'SPLS': SPLS_fitting,
            'EN': EN_fitting,
            'LASSO': LASSO_fitting,
            'ALVEN': ALVEN_fitting,
            'RR': RR_fitting,
            'DALVEN': DALVEN_fitting,
            'DALVEN_full_nonlinear': DALVEN_fitting_full_nonlinear}
    return switcher[model_name]

def OLS_fitting(X, y, X_test, y_test):
    """
    Fits data using ordinary least squares: y = a*x1 + b*x2 + ...

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    """

    # Training
    OLS_model = sm.OLS(y, X).fit()
    yhat_train = OLS_model.predict().reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    OLS_params = OLS_model.params.reshape(-1,1) # Fitted parameters
    
    # Testing
    yhat_test = OLS_model.predict(X_test).reshape((-1,1))
    mse_test = MSE(y_test, yhat_test)
    return(OLS_model, OLS_params, mse_train, mse_test, yhat_train, yhat_test)

def SPLS_fitting(X, y, X_test, y_test, K = None, eta = None, eps = 1e-4, maxstep = 1000):
    """
    Fits data using an Sparse PLS model (see doi.org/10.1111%2Fj.1467-9868.2009.00723.x)

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    K: int, optional, default = None
        Number of latent variables
    eta: float, optional, default = None
        Sparsity tuning parameter ranging from 0 to 1
    """
    _, selected_variables, _, _ = SPLS(X, y, K, eta, eps = eps, max_steps = maxstep)
    if len(selected_variables) >= K:
        SPLS_model = PLSRegression(K, scale = False, tol = eps).fit(X[:, selected_variables], y)
    else:
        return None, None, np.inf, np.inf, None, None # TODO: should we really return np.inf?
    SPLS_params = SPLS_model.coef_.squeeze()
    # Predictions and MSEs
    yhat_train = np.dot(X[:, selected_variables], SPLS_params)
    yhat_test = np.dot(X_test[:, selected_variables], SPLS_params)
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test

def EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter = 10000, tol = 1e-4, use_cross_entropy = False):
    """
    Fits data using sklearn's Elastic Net model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    """

    # Training
    EN_model = ElasticNet(random_state = 0, alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = max_iter, tol = tol)
    EN_model.fit(X, y)
    yhat_train = EN_model.predict(X).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    EN_params = EN_model.coef_.reshape((-1,1)) # Fitted parameters

    # Testing
    yhat_test = EN_model.predict(X_test).reshape((-1,1))
    mse_test = MSE(y_test, yhat_test)
    return (EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test)

def RR_fitting(X, y, X_test, y_test, alpha, l1_ratio):
    """
    Fits data using sklearn's Ridge Regression model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    """

    # Training
    RR_model = Ridge(alpha = alpha, fit_intercept = False).fit(X, y)
    yhat_train = np.dot(X, RR_params).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    RR_params = RR_model.coef_.reshape((-1,1)) # Fitted parameters

    # Testing
    yhat_test = np.dot(X_test, RR_params).reshape((-1,1))
    mse_test = MSE(y_test, yhat_test)
    return (RR_model, RR_params, mse_train, mse_test, yhat_train, yhat_test)

def LASSO_fitting(X, y, X_test, y_test, alpha, max_iter = 10000, tol = 1e-4):
    """
    Fits data using sklearn's LASSO model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        Regularization parameter weight.
    """

    # Training
    LASSO_model = Lasso(random_state = 0, alpha = alpha, fit_intercept = False, max_iter = max_iter, tol = tol)
    LASSO_model.fit(X, y)
    yhat_train = LASSO_model.predict(X).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    LASSO_params = LASSO_model.coef_.reshape((-1,1)) # Fitted parameters

    # Testing
    yhat_test = LASSO_model.predict(X_test).reshape((-1,1))
    mse_test = MSE(y_test, yhat_test)
    return (LASSO_model, LASSO_params, mse_train, mse_test, yhat_train, yhat_test)

def ALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree = 1, alpha_num = None, cv = False, max_iter = 10000, 
                tol = 1e-4, selection = 'p_value', select_value = 0.15, trans_type = 'auto', use_cross_entropy = False):
    """
    Fits data using Algebraic Learning Via Elastic Net
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    degree : int, optional, default = 1
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'auto'
        Feature transformation based on ALVEN ('auto') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """

    # Feature transformation
    if trans_type == 'auto':
        X, X_test = nr.feature_trans(X, X_test, degree = degree, interaction = 'later')
    elif trans_type == 'poly':
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
    else:
        raise ValueError(f'trans_type must be "auto" or "poly", but you passed {trans_type}')

    # Remove features with insignificant variance
    sel = VarianceThreshold(threshold = tol).fit(X)
    X = sel.transform(X)
    X_test = sel.transform(X_test)

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    X_test = scaler_x.transform(X_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)
  
    # Eliminate features
    f_test, p_values = f_regression(X, y.flatten())

    if selection == 'p_value':
        retain_index = p_values < select_value
    elif selection == 'percentage':
        number = int(np.ceil(select_value * X.shape[1]))
        f_test.sort()
        value = f_test[-number]
        retain_index = f_test >= value
    else:
        f = np.copy(f_test)
        f = np.sort(f)[::-1]

        axis = np.linspace(0, len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1), f.reshape(-1,1)), axis = 1)

        lineVec = AllCord[-1] - AllCord[0]
        lineVec /= np.sqrt(np.sum(lineVec**2))
        vecFromFirst = AllCord- AllCord[0] # Distance from each point to the line
        # And calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis = 1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine**2, axis = 1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        retain_index = f_test>=value

    X_fit =  X[:, retain_index]
    X_test_fit = X_test[:, retain_index]

    if X_fit.shape[1] == 0:
        print('No variable was selected by ALVEN')
        ALVEN_model = None
        ALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            X_max = np.concatenate((X_fit,X_test_fit),axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(X_max.T,y_max) ** 2, axis=1)).max())/X_max.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(X_fit.T,y) ** 2, axis=1)).max())/X_fit.shape[0]/l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        # EN for model fitting
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(X_fit, y, X_test_fit, y_test, alpha, l1_ratio, max_iter, tol, use_cross_entropy)
    return (ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index)

def DALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, alpha_num = None, cv = False, max_iter = 10000, 
                tol = 1e-4, selection = 'p_value', select_value = 0.10, trans_type = 'auto', use_cross_entropy = False):
    """
    Fits data using Dynamic Algebraic Learning Via Elastic Net
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    lag : int
        Variable lags to be considered. A lag of L will make the model take into account ...
            variables from point xt to xt-L and from yt to yt-L
    degree : int
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'auto'
        Feature transformation based on ALVEN ('auto') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """

    # Feature transformation
    if trans_type == 'auto':
        X, X_test = nr.feature_trans(X, X_test, degree = degree, interaction = 'later')
    elif trans_type == 'poly':
        X, X_test = nr.poly_feature(X, X_test, degree = degree, interaction = True, power = True)
    else:
        raise ValueError(f'trans_type must be "auto" or "poly", but you passed {trans_type}')

    # Lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD, X[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, X_test[lag-1-i : -i-1]))
    # Lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD, y[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, y_test[lag-1-i : -i-1]))    
    # Shorterning y
    y = y[lag:]
    y_test = y_test[lag:]

    # Remove features with insignificant variance
    sel = VarianceThreshold(threshold = tol).fit(XD)
    XD = sel.transform(XD)
    XD_test = sel.transform(XD_test)

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)

    # Eliminate features
    f_test, p_values = f_regression(XD, y.flatten())

    if selection == 'p_value':
        retain_index = p_values < select_value
    elif selection == 'percentage':
        number = int(np.ceil(select_value * XD.shape[1]))
        f_test.sort()
        value = f_test[-number]
        retain_index = f_test >= value
    else:
        f = np.copy(f_test)
        f = np.sort(f)[::-1]

        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1), f.reshape(-1,1)), axis = 1)

        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        vecFromFirst = AllCord- AllCord[0] # Distance from each point to the line
        # And calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis = 1) # np.repeat(np.atleast_2d(lineVec), len(f), 0)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine**2, axis = 1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        retain_index = f_test >= value

    XD_fit =  XD[:, retain_index]
    XD_test_fit = XD_test[:, retain_index]

    if XD_fit.shape[1] == 0:
        print('No variable was selected by ALVEN')
        DALVEN_model = None
        DALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            XD_max = np.concatenate((XD_fit, XD_test_fit), axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(XD_max.T, y_max)**2, axis = 1)).max()) / XD_max.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(XD_fit.T, y) ** 2, axis=1)).max()) / XD_fit.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        #EN for model fitting
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(XD_fit, y, XD_test_fit, y_test, alpha, l1_ratio, max_iter, tol, use_cross_entropy)

        num_train = XD_fit.shape[0]
        num_parameter = sum(DALVEN_params!=0)[0]
        AIC = num_train*np.log(mse_train) + 2*num_parameter
        AICc = num_train*np.log(mse_train) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train) # TODO: Fix the divide by zero errors
        BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    return (DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, (AIC,AICc,BIC))

def DALVEN_fitting_full_nonlinear(X, y, X_test, y_test, alpha, l1_ratio, degree, lag, alpha_num = None, cv = False, max_iter = 10000, 
                                tol = 1e-4, selection = 'p_value', select_value = 0.10, trans_type = 'auto', use_cross_entropy = False):
    """
    Fits data using Dynamic Algebraic Learning Via Elastic Net - full non-linear mapping
    https://doi.org/10.1016/j.compchemeng.2020.107103

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float or int
        Regularization parameter weight.
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    lag : int
        Variable lags to be considered. A lag of L will make the model take into account ...
            variables from point xt to xt-L and from yt to yt-L
    degree : int
        The degrees of nonlinear mapping.
    alpha_num : int, optional, default = None
        Penalty weight used.
    cv : bool, optional, default = False
        Whether the run is done to validate a model or test the best model.
    selection : str, optional, default = 'p_value'
        Selection criterion for the pre-processing step
        Must be in {'p_value', 'percentage', 'elbow'}
    select_value : float, optional, default = 0.10
        The minimum p_value for a variable to be considered relevant (when selection == 'p_value'), ...
            or the first select_value percent variables to be used (when selection == 'percentage').
        Not relevant when selection == 'elbow'
    trans_type : str, optional, default = 'auto'
        Feature transformation based on ALVEN ('auto') or polynomial ('poly')
    use_cross_entropy : bool, optional, default = False
        Whether to use cross entropy or MSE for model comparison.
    """
    # Lag padding for X
    XD = X[lag:]
    XD_test = X_test[lag:]
    for i in range(lag):
        XD = np.hstack((XD, X[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, X_test[lag-1-i : -i-1]))
    # Lag padding for y in design matrix
    for i in range(lag):
        XD = np.hstack((XD, y[lag-1-i : -i-1]))
        XD_test = np.hstack((XD_test, y_test[lag-1-i : -i-1]))
    # Shorterning y
    y = y[lag:]
    y_test = y_test[lag:]

    # Feature transformation
    if trans_type == 'auto':
        XD, XD_test = nr.feature_trans(XD, XD_test, degree = degree, interaction = 'later')
    else:
        XD, XD_test = nr.poly_feature(XD, XD_test, degree = degree, interaction = True, power = True)
  
    # Remove features with insignificant variance
    sel = VarianceThreshold(threshold = tol).fit(XD)
    XD = sel.transform(XD)
    XD_test = sel.transform(XD_test)

    # Scale data (based on z-score)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(XD)
    XD = scaler_x.transform(XD)
    XD_test = scaler_x.transform(XD_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)

    # Eliminate features
    f_test, p_values = f_regression(XD, y.flatten())

    if selection == 'p_value':
        retain_index = p_values < select_value
    elif selection == 'percentage':
        number = int(np.ceil(select_value * XD.shape[1]))
        f_test.sort()
        value = f_test[-number]
        retain_index = f_test >= value
    else:
        f = np.copy(f_test)
        f = np.sort(f)[::-1]

        axis = np.linspace(0,len(f)-1, len(f))
        AllCord = np.concatenate((axis.reshape(-1,1), f.reshape(-1,1)), axis = 1)

        lineVec = AllCord[-1] - AllCord[0]
        lineVec = lineVec/ np.sqrt(np.sum(lineVec**2))
        vecFromFirst = AllCord- AllCord[0] # Distance from each point to the line
        # And calculate the distance of each point to the line
        scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVec, len(f), 1), axis = 1) # np.repeat(np.atleast_2d(lineVec), len(f), 0)
        vecFromFirstParallel = np.outer(scalarProduct, lineVec)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine**2, axis = 1))
        BestPoint = np.argmax(distToLine)
        value = f[BestPoint]
        retain_index = f_test >= value
 
    XD_fit =  XD[:, retain_index]
    XD_test_fit = XD_test[:, retain_index]

    if XD_fit.shape[1] == 0:
        print('No variable was selected by ALVEN')
        DALVEN_model = None
        DALVEN_params = None
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape)
        yhat_test = np.zeros(y_test.shape)
        alpha = 0
    else:
        if alpha_num is not None and cv:
            XD_max = np.concatenate((XD_fit, XD_test_fit), axis = 0)
            y_max = np.concatenate((y, y_test), axis = 0)
            alpha_max = (np.sqrt(np.sum(np.dot(XD_max.T,y_max)**2, axis = 1)).max()) / XD_max.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        if alpha_num is not None and not cv:
            alpha_max = (np.sqrt(np.sum(np.dot(XD_fit.T, y)**2, axis = 1)).max()) / XD_fit.shape[0] / l1_ratio
            alpha_list = np.logspace(np.log10(alpha_max * tol), np.log10(alpha_max), alpha_num)[::-1]
            alpha = alpha_list[alpha]

        #EN for model fitting
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(XD_fit, y, XD_test_fit, y_test, alpha, l1_ratio, max_iter, tol, use_cross_entropy)

        num_train = XD_fit.shape[0]
        num_parameter = sum(DALVEN_params!=0)[0]
        AIC = num_train*np.log(mse_train) + 2*num_parameter
        AICc = num_train*np.log(mse_train) + (num_parameter+num_train)/(1-(num_parameter+2)/num_train)
        BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    return (DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index, (AIC,AICc,BIC))

