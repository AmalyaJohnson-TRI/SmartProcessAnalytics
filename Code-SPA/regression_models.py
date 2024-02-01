import statsmodels.api as sm
from SPLS_Python import SPLS
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import warnings
warnings.filterwarnings("ignore") # TODO: Want to just ignore the PLS constant residual warnings, but this will do for now

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

def PLS_fitting(X, y, X_test, y_test, K = None, eps = 1e-4, maxstep = 1000): # TODO: merge with SPLS_fitting, since this is just a special case with eta = 0
    """
    Fits data using a PLS model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    K: int, optional, default = None
        Number of latent variables
    """
    PLS_model = PLSRegression(K, scale = False, tol = eps).fit(X, y)
    PLS_params = PLS_model.coef_.squeeze()
    # Predictions and MSEs
    yhat_train = np.dot(X, PLS_params)
    yhat_test = np.dot(X_test, PLS_params)
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return PLS_model, PLS_params, mse_train, mse_test, yhat_train, yhat_test

def EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter = 10000, tol = 1e-4):
    """
    Fits data using sklearn's Elastic Net model

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        The weight of the L1 and L2 regularizations: alpha*l1_ratio*||w||_1 + 0.5*alpha*(1 - l1_ratio)*||w||^2_2
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    """
    EN_model = ElasticNet(random_state = 0, alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = max_iter, tol = tol)
    EN_model.fit(X, y)
    EN_params = EN_model.coef_ # Fitted parameters
    # Predictions and MSEs
    yhat_train = EN_model.predict(X).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    yhat_test = EN_model.predict(X_test).reshape((-1,1))
    mse_test = MSE(y_test, yhat_test)
    return (EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test)

def ALVEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree = 1, lag = 0, max_iter = 10000, tol = 1e-4, trans_type = 'all', ALVEN_type = 'ALVEN', interaction = True, selection = None):
    """
    Fits data using Algebraic Learning Via Elastic Net (ALVEN), a nonlinear expansion of Elastic Net.
    ALVEN was originally published in doi.org/10.1016/j.compchemeng.2020.107103
    The implementation below uses the LCEN algorithm over the original ALVEN algorithm.
    LCEN is an upgraded, non-linear feature selection algorithm with considerably improved feature selection capabilities, model accuracy, and speed.

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    alpha : float
        The weight of the L1 and L2 regularizations: alpha*l1_ratio*||w||_1 + 0.5*alpha*(1 - l1_ratio)*||w||^2_2
    l1_ratio : float
        Ratio of L1 penalty to total penalty. When l1_ratio == 1, only the L1 penalty is used.
    degree : int, optional, default = 1
        The degrees of nonlinear mapping.
    lag : int, optional, default = 0
        Variable lags to be considered. A lag of L will make the model take into account ...
            variables from point xt to xt-L and from yt to yt-L.
        Relevant only when ALVEN_type != 'alven'.
    trans_type : str in {'all', 'poly', 'simple_interaction'}, optional, default == 'all'
        Whether to include all transforms (polynomial, log, sqrt, and inverse), only polynomial transforms (and, ...
            optionally, interactions), or just interactions.
        The log, sqrt, and inverse terms never include interactions among the same transform type (such as ln(x1)*ln(x4)), ...
            but include some interactions among each other for the same variable (such as ln(x0)/x0, x0^1.5, etc.).
    ALVEN_type : string in {'alven', 'dalven', 'dalven_full_nonlinear'}, optional, default = 'alven'
        Whether to use the ALVEN, the regular DALVEN or DALVEN full nonlinear. The difference is whether the variables representing ...
              previous time points are included in the nonlinear transformations (DALVEN_full_nonlinear) or not (DALVEN)
    interaction : bool, optional, default = True
        Whether to also include interactions between different variables (such as x1*x3 or x0*x5^2)
        Note that, if trans_type == 'simple_interaction', this variable must be set to True
    selection : array of Bool, optional, default = None
        Which variables will be used. This is obtained automatically by SPA and should not be passed by the user
    """
    if ALVEN_type.casefold() in {'alven', 'dalven'} and X.shape[1] > 0: # Feature transformation; done below for DALVEN_type == 'dalven_full_nonlinear'
        X, X_test, label_names = _feature_trans(X, X_test, degree, interaction, trans_type)
    elif ALVEN_type.casefold() in {'alven', 'dalven'}: # Just the intercept - that is, an Nx1 vector full of 1
        label_names = ['intercept']
        X = np.ones((X.shape[0], 1))
        X_test = np.ones((X_test.shape[0], 1))
    # Adding variables from previous timesteps. Not relevant for ALVEN
    if ALVEN_type.casefold() in {'dalven', 'dalven_full_nonlinear'}:
        not_intercept_idx = 1 if ALVEN_type.casefold() == 'dalven' else 0 # We ignore the 1st column because it is the intercept after X goes through _feature_trans()
        # Lag padding for X
        X_temp = np.hstack([X[lag-1-idx : -idx-1, not_intercept_idx:] for idx in range(lag)]) # The additional entries representing the previous times (t-1 to t-lag)
        y_temp = np.hstack([y[lag-1-idx : -idx-1] for idx in range(lag)])
        X = np.hstack((X[lag:], X_temp, y_temp))
        X_test_temp = np.hstack([X_test[lag-1-idx : -idx-1, not_intercept_idx:] for idx in range(lag)])
        y_test_temp = np.hstack([y_test[lag-1-idx : -idx-1] for idx in range(lag)])
        X_test = np.hstack((X_test[lag:], X_test_temp, y_test_temp))
        if ALVEN_type.casefold() == 'dalven':
            label_names = np.concatenate((label_names, [elem + f'(t-{idx+1})' for idx in range(lag) for elem in label_names[1:]], [f'y(t-{idx+1})' for idx in range(lag)]))
        # Shorterning y
        y = y[lag:]
        y_test = y_test[lag:]
    if ALVEN_type.casefold() == 'dalven_full_nonlinear': # Feature transformation; done above for ALVEN_type in {'alven', 'dalven'}
        X, X_test, label_names = _feature_trans(X, X_test, degree, interaction, trans_type)

    # Scale data (mean = 0, stdev = 1)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    X_test = scaler_x.transform(X_test)
    scaler_y = StandardScaler(with_mean=True, with_std=True)
    scaler_y.fit(y)
    y = scaler_y.transform(y)
    y_test = scaler_y.transform(y_test)

    if selection is not None:
        X = X[:, selection]
        X_test = X_test[:, selection]

    if X.shape[1] == 0:
        ALVEN_model = None
        ALVEN_params = np.empty(0)
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape) # Ok to be 0 because y is scaled --> mean(y) is approximately 0
        yhat_test = np.zeros(y_test.shape)
    else:
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter, tol)
    # Calculating information criteria values
    num_train = X.shape[0] # Typically written as n
    num_parameter = (ALVEN_params!=0).flatten().sum() # Typically written as k
    AIC = num_train*np.log(mse_train) + 2*num_parameter # num_train * log(MSE) is one of the formulae to replace L. Shown in https://doi.org/10.1002/wics.1460, for example.
    AICc = AIC + 2*num_parameter*(num_parameter+1) / (num_train - num_parameter - 1)
    BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    return (ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, label_names, (AIC, AICc, BIC))

def RF_fitting(X, y, X_test, y_test, n_estimators = 100, max_depth = 10, min_samples_leaf = 0.1, max_features = 1.0, random_state = 0):
    """
    Fits data using a random forest regressor

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    n_estimators : int, optional, default = 100
        Number of trees in the forest.
    max_depth : int, optional, default = 10
        Maximum depth of a single tree.
    min_samples_leaf : int or (float < 1), optional, default = 0.1
        The minimum number of samples per leaf.
        If float, represents a fraction of the samples used as the minimum.
    max_features : int, (float < 1), None, or str in {'sqrt', 'log2'}, optional, default = 1.0
        Maximum number of features when considered for a potential splitting.
        If float, represents a fraction of the features used as the minimum.
        Setting it to None, 'auto', or 1.0 uses all available features (X.shape[1] = m).
    random_state : int or None, optional, default = 0
        Seed for the random number generator.
    """
    RF = RandomForestRegressor(n_estimators, max_depth = max_depth, random_state = random_state,
                                    max_features = max_features, min_samples_leaf = min_samples_leaf)
    RF.fit(X, y.flatten())
    # Predictions and MSEs
    yhat_train = RF.predict(X).reshape((-1,1))
    yhat_test = RF.predict(X_test).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return (RF, mse_train, mse_test, yhat_train, yhat_test)

def SVR_fitting(X, y, X_test, y_test, C = 100, epsilon = 10, gamma = 'auto', tol = 1e-4, max_iter = 10000):
    """
    Support Vector Machine-based regression

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    C : float > 0, optional, default = 100
        Parameter inversely proportional to the strength of the regularization
    epsilon : float >= 0, optional, default = 10
        Epsilon-tube within which no penalty is associated in the training loss function
    gamma : (float >=0) or str in {'scale', 'auto'}, optional, default = 'auto'
        Kernal coefficient for the 'rbf', 'poly', and 'sigmoid' kernels
        If 'auto', gamma = 1 / X.shape[1]
    """
    SVR_model = SVR(gamma = gamma, C = C, epsilon = epsilon, tol = tol, max_iter = max_iter)
    SVR_model.fit(X, y.flatten())
    # Predictions and MSEs
    yhat_train = SVR_model.predict(X).reshape((-1,1))
    yhat_test = SVR_model.predict(X_test).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return (SVR_model, mse_train, mse_test, yhat_train, yhat_test)

def _feature_trans(X, X_test = None, degree = 2, interaction = True, trans_type = 'all'):
    """
    A helper function that is automatically called by SPA.
    Performs non-linear transformations of X (and X_test). Transformations include polynomial transforms up to "degree", ...
    interactions between the raw variables in X (up to "degree" powers at the same time). If trans_type == 'all', also ...
    includes ln, sqrt, and inverse transforms of power up to "degree". Also includes some interaction terms among them.

    Parameters
    ----------
    X, y : Numpy array with shape N x m
        Training data predictors.
    X_test, y_test : Numpy array with shape N_test x m, optional, default = None
        Testing data predictors.
    degree : integer, optional, default = 2
        The highest degree used for polynomial and interaction transforms.
        For example, degree = 1 would include x0 to xN. Degree = 2 would also include (x0)^2 to (xN)^2...
            and, if interaction == True, (x0*x1), (x0*x2), ..., (x0*xN), ..., (x1*xN) terms.
    interaction : boolean, optional, default = True
        Whether to include polynomial interaction terms up to "degree". For example, degree = 2 interactions include ...
            terms of the x0*x1 form. Degree = 3 interactions also include terms of the x2*x3*x5 and (x1)^2 * x4 form
    trans_type : str in {'all', 'poly', 'simple_interaction'}, optional, default == 'all'
        Whether to include all transforms (polynomial, log, sqrt, and inverse), only polynomial transforms (and, ...
            optionally, interactions), or just interactions.
        The log, sqrt, and inverse terms never include interactions among the same transform type (such as ln(x1)*ln(x4)), but ...
            include some interactions among each other for the same variable (such as ln(x0)*1/x0, x0*sqrt(x0), etc.).
    """
    X_test_out = None # Declaring this variable to avoid errors when doing the return statement

    # Polynomial transforms (and interaction terms)
    poly = PolynomialFeatures(degree, include_bias = True, interaction_only = trans_type.casefold() == 'simple_interaction')
    X_out = poly.fit_transform(X)
    label_names = poly.get_feature_names_out()
    interaction_column = np.array([' ' in elem for elem in label_names], dtype = bool) # To filter out interactions if user asked for a polynomial-only transform. Also for the log, sqrt, and inv terms below when poly_trans_only == False
    for idx in range(len(label_names)):
        label_names[idx] = label_names[idx].translate({ord(i): '*' for i in ' '}) # str.translate replaces the characters on the right of the for (in this case, a whitespace) with an asterisk
    if X_test is not None:
        X_test_out = poly.fit_transform(X_test)
    # Discarding the interaction terms (x1*x2, x2*x3*x5, (x1)^2 * x4, etc.) if requested to do so by the user
    if not interaction:
        X_out = X_out[:, ~interaction_column]
        label_names = label_names[~interaction_column]
        if X_test is not None:
            X_test_out = X_test_out[:, ~interaction_column]
    # Including ln, sqrt, and inverse terms; and also their higher-degree transforms if degree >= 2
    if trans_type.casefold() == 'all':
        # Setting up the transforms
        Xlog = np.where(X!=0, np.log(np.abs(X)), -50) # Avoiding log(0) = -inf
        if X_test is not None: # Columns where all entries are >= 0 for sqrt
            all_pos = np.all(X >= 0, axis = 0) & np.all(X_test >= 0, axis = 0)
        else:
            all_pos = np.all(X >= 0, axis = 0)
        Xsqrt = np.sqrt(X[:, all_pos])
        Xinv = 1/X
        Xinv[Xinv == np.inf] = 1e15
        Xinv[Xinv == -np.inf] = -1e15
        if X_test is not None:
            Xlog_t = np.where(X_test!=0, np.log(np.abs(X_test)), -50) # Avoiding log(0) = -inf
            Xsqrt_t = np.sqrt(X_test[:, all_pos])
            Xinv_t = 1/X_test
            Xinv_t[Xinv_t == np.inf] = 1e15
            Xinv_t[Xinv_t == -np.inf] = -1e15
        # ln transform
        Xlog_trans = poly.fit_transform(Xlog)[:, ~interaction_column][:, 1:] # Transforming and removing interactions, as we do not care about log-log interactions
        X_out = np.column_stack((X_out, Xlog_trans))
        temp_label_names = poly.get_feature_names_out()[~interaction_column][1:] # [1:] to remove the bias term, as it was already included with the polynomial transformations
        for idx in range(len(temp_label_names)): # Converting the names from x0-like to ln(x0)-like
            power_split = temp_label_names[idx].split('^') # Separates the variable (e.g.: x0 or x1) from the power it was raised to, if it exists
            if len(power_split) > 1:
                power = f'^{power_split[-1]}'
                base = ''.join(power_split[:-1])
            else: # Variable hasn't beed raised to any power (equivalent to ^1), but we need an empty "power" variable to avoid errors
                power = ''
                base = power_split[0]
            temp_label_names[idx] = f'ln({base}){power}' # Final name is of the form ln(x1)^3, not ln(x1^3)
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            Xlog_test_trans = poly.fit_transform(Xlog_t)[:, ~interaction_column][:, 1:]
            X_test_out = np.column_stack((X_test_out, Xlog_test_trans))
        # sqrt transform
        X_out = np.column_stack((X_out, Xsqrt))
        temp_label_names = [f'sqrt(x{number})' for number in range(X.shape[1]) if all_pos[number]]
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            X_test_out = np.column_stack((X_test_out, Xsqrt_t))
        # Inverse transform
        Xinv_trans = poly.fit_transform(Xinv)[:, 1:] # [:, 1:] to remove the bias term, as it was already included with the polynomial transformations
        X_out = np.column_stack((X_out, Xinv_trans))
        temp_label_names = poly.get_feature_names_out()[1:] # [1:] to remove the bias term, as it was already included with the polynomial transformations
        for idx in range(len(temp_label_names)): # Converting the names from x0-like to ln(x0)-like
            temp_label_names[idx] = temp_label_names[idx].translate({ord(i): '*' for i in ' '}) # str.translate replaces the characters on the right of the for (in this case, a whitespace) with an asterisk
            temp_label_names[idx] = f'1/({temp_label_names[idx]})' # 1/(x1^3) is the same as (1/x1)^3, so we do not need the fancy manipulations used above in the ln transform naming
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            Xinv_test_trans = poly.fit_transform(Xinv_t)[:, 1:]
            X_test_out = np.column_stack((X_test_out, Xinv_test_trans))
        # Specific interactions between X, ln(X), sqrt(X), and 1/X that occur for degree >= 2
        if degree >= 2:
            normal_plus_half_interaction = np.column_stack([X[:, all_pos]**(pow1+0.5) for pow1 in range(1, degree)])
            normal_plus_half_names = [f'x{number}^{pow1+0.5}' for pow1 in range(1, degree) for number in range(X.shape[1]) if all_pos[number]]
            log_inv_interaction = np.column_stack([Xlog**pow1 * Xinv**pow2 for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree])
            log_inv_names = [f'ln(x{number})^{pow1}/(x{number})^{pow2}' for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree for number in range(X.shape[1])]
            log_inv_names = [elem[:-2].replace('^1/', '/') + elem[-2:].replace('^1', '') for elem in log_inv_names] # Removing ^1. String addition to avoid removing ^10, ^11, ^12, ...
            inv_minus_half_interaction = np.column_stack([Xinv[:, all_pos]**(pow1-0.5) for pow1 in range(1, degree)])
            inv_minus_half_names = [f'1/(x{number}^{pow1-0.5})' for pow1 in range(1, degree) for number in range(X.shape[1]) if all_pos[number]]
            if degree == 2: # degree == 2 does not have the ln(x) * sqrt(x) / x type of interactions
                X_out = np.column_stack((X_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction))
                label_names = np.concatenate((label_names, normal_plus_half_names, log_inv_names, inv_minus_half_names))
            else:
                log_inv_minus_oneandhalf_interaction = np.column_stack([Xlog[:, all_pos]**pow1 * Xinv[:, all_pos]**(pow2-0.5) for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree-1])
                log_inv_minus_oneandhalf_names = [f'ln(x{number})^{pow1}/(x{number}^{pow2-0.5})' for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree-1 for number in range(X.shape[1]) if all_pos[number]]
                log_inv_minus_oneandhalf_names = [elem[:-2].replace('^1/', '/') + elem[-2:].replace('^1', '') for elem in log_inv_minus_oneandhalf_names] # Removing ^1. String addition to avoid removing ^10, ^11, ^12, ...
                X_out = np.column_stack((X_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction, log_inv_minus_oneandhalf_interaction))
                label_names = np.concatenate((label_names, normal_plus_half_names, log_inv_names, inv_minus_half_names, log_inv_minus_oneandhalf_names))
            if X_test is not None:
                normal_plus_half_interaction = np.column_stack([X_test[:, all_pos]**(pow1+0.5) for pow1 in range(1, degree)])
                log_inv_interaction = np.column_stack([Xlog_t**pow1 * Xinv_t**pow2 for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree])
                inv_minus_half_interaction = np.column_stack([Xinv_t[:, all_pos]**(pow1-0.5) for pow1 in range(1, degree)])
                if degree == 2:
                    X_test_out = np.column_stack((X_test_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction))
                else:
                    log_inv_minus_oneandhalf_interaction = np.column_stack([Xlog_t[:, all_pos]**pow1 * Xinv_t[:, all_pos]**(pow2-0.5) for pow1 in range(1,degree) for pow2 in range(1,degree) if pow1+pow2 <= degree-1])
                    X_test_out = np.column_stack((X_test_out, normal_plus_half_interaction, log_inv_interaction, inv_minus_half_interaction, log_inv_minus_oneandhalf_interaction))

    return (X_out, X_test_out, label_names)
