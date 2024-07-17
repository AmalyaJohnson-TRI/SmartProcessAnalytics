import statsmodels.api as sm
from SPLS_Python import SPLS
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
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
    yhat_train = OLS_model.predict()
    mse_train = MSE(y, yhat_train)
    OLS_params = OLS_model.params.reshape(-1,1) # Fitted parameters
    
    # Testing
    yhat_test = OLS_model.predict(X_test)
    mse_test = MSE(y_test, yhat_test)
    return(OLS_model, OLS_params, mse_train, mse_test, yhat_train, yhat_test)

def SPLS_fitting(X, y, X_test, y_test, K = None, eta = None, eps = 1e-4, maxstep = 1000):
    """
    Fits data using a Sparse PLS model (see doi.org/10.1111%2Fj.1467-9868.2009.00723.x)

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    K: int, optional, default = None
        Number of latent variables
    eta: float, optional, default = None
        Sparsity tuning parameter ranging from 0 to 1. 0 is equivalent to PLS.
    """
    if eta:
        _, selected_variables, _, _ = SPLS(X, y, K, eta, eps = eps, max_steps = maxstep)
    else: # eta = 0 is equivalent to regular PLS, which selects all variables
        selected_variables = np.ones(X.shape[1], dtype = bool)
    if len(selected_variables) >= K:
        SPLS_model = PLSRegression(K, scale = False, tol = eps).fit(X[:, selected_variables], y)
    else:
        return None, None, np.inf, np.inf, None, None
    SPLS_params = SPLS_model.coef_.squeeze()
    # Predictions and MSEs
    yhat_train = np.dot(X[:, selected_variables], SPLS_params)
    yhat_test = np.dot(X_test[:, selected_variables], SPLS_params)
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test

def EN_fitting(X, y, X_test, y_test, alpha, l1_ratio, max_iter = 10000, tol = 1e-4, random_state = 0):
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
    max_iter : int, optional, default = 10000
        The maximum number of iterations
    random_state : int or None, optional, default = 0
        Seed for the random number generator.
    """
    EN_model = ElasticNet(random_state = random_state, alpha = alpha, l1_ratio = l1_ratio, fit_intercept = False, max_iter = max_iter, tol = tol)
    EN_model.fit(X, y)
    EN_params = EN_model.coef_ # Fitted parameters
    # Predictions and MSEs
    yhat_train = EN_model.predict(X).reshape((-1,1))
    mse_train = MSE(y, yhat_train)
    yhat_test = EN_model.predict(X_test).reshape((-1,1))
    mse_test = MSE(y_test, yhat_test)
    return (EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test)

def LCEN_fitting(X, y, X_test, y_test, alpha, l1_ratio, degree = 1, lag = 0, min_lag = 0, trans_type = 'all', interaction = True, selection = None, transform_y = False, all_pos_X = True, all_pos_y = True):
    """
    Fits data using the LASSO-Clip-EN (LCEN) algorithm (doi.org/10.48550/arXiv.2402.17120).
    LCEN is a powerful, non-linear, and interpretable feature selection algorithm with considerable feature selection capabilities, model accuracy, and speed.
    LCEN was created as an upgrade to ALVEN, which was originally published in doi.org/10.1016/j.compchemeng.2020.107103

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
    min_lag : int < lag, optional, default = 0
        The minimum lag used in autoregressive LCEN. Should be smaller than the smallest lag.
        Useful when making N-points ahead predictions, in which case min_lag should be set to N-1 and ...
            each entry in lag should be increased by N-1.
            (e.g.: the default min_lag = 0 is used to make predictions 1 point ahead; a min_lag = 5 is
            used to make predictions 6 points ahead, ignoring the first 5 points ahead when training.)
    trans_type : str in {'all', 'poly', 'simple_interaction'}, optional, default == 'all'
        Whether to include all transforms (polynomial, log, sqrt, and inverse), only polynomial transforms (and, ...
            optionally, interactions), or just interactions.
        The log, sqrt, and inverse terms never include interactions among the same transform type (such as ln(x1)*ln(x4)), ...
            but include some interactions among each other for the same variable (such as ln(x0)/x0, x0^1.5, etc.).
    interaction : bool, optional, default = True
        Whether to also include interactions between different variables (such as x1*x3 or x0*x5^2)
        Note that, if trans_type == 'simple_interaction', this variable must be set to True
    selection : array of Bool, optional, default = None
        Which variables will be used. This is obtained automatically by SPA and should not be passed by the user
    transform_y : Bool, optional, default = False
        Whether to also perform transforms on the y measurements from previous timepoints
        Relevant only when lag > 0
    all_pos_X, all_pos_y : bool, optional, default = True
        Whether all entries in the X/y data are positive or also contain some negative entries. This is done to avoid taking ...
            sqrt(k) for some k<0 when the features are transformed/expanded.
        This used to be determined within _feature_trans(), but it leads to array size problems when some folds have negative ...
            entries and some do not.
    """
    # Expanding the X features
    if not transform_y and X.shape[1] > 0:
        X, X_test, label_names = _feature_trans(X, X_test, degree, interaction, trans_type, all_pos_X, all_pos_y)
    elif X.shape[1] == 0:
        label_names = np.array([])
    if transform_y:
        n_X_vars = X.shape[1]
    # Adding variables from previous timesteps (t-1 to t-lag)
    if lag > 0:
        X_temp = np.hstack([X[lag-1-idx : -idx-1, :] for idx in range(min_lag, lag)]) # The additional entries representing the previous times (t-1 to t-lag)
        y_temp = np.hstack([y[lag-1-idx : -idx-1] for idx in range(min_lag, lag)])
        X_test_temp = np.hstack([np.concatenate((X[X.shape[0]-idx-1:, :], X_test[: -idx-1, :])) for idx in range(min_lag, lag)]) # No need to remove entries from X_test or y_test because we can use the data from the final points of X or y to predict the initial points of X_test or y_test
        y_test_temp = np.hstack([np.concatenate((y[(len(y)-idx-1):],  y_test[: -idx-1])) for idx in range(min_lag, lag)])
        X = np.hstack((X[lag:], X_temp, y_temp))
        y = y[lag:]
        X_test = np.hstack((X_test, X_test_temp, y_test_temp))
        if not transform_y:
            label_names = np.concatenate((label_names, [elem + f'(t-{idx+1})' for idx in range(lag-min_lag) for elem in label_names], [f'y(t-{idx+1})' for idx in range(lag-min_lag)]))
    if transform_y: # Feature transformation that includes the y features in the X and X_test variables
        X, X_test, label_names = _feature_trans(X, X_test, degree, interaction, trans_type, all_pos_X, all_pos_y)
        # Correcting the label names
        for idx in range(n_X_vars*(lag+1) + lag - 1, n_X_vars-1, -1): # xN to x(n_X_vars). In reverse order to avoid x11 to x19 being caught in x1, for example
            for elem_idx, elem in enumerate(label_names):
                if idx >= n_X_vars * (lag + 1): # These entries are the y(t-1), y(t-2), ..., y(t-N) entries
                    replacement = f'y(t-{idx - n_X_vars * (lag+1) + 1})'
                elif idx >= n_X_vars: # These entries are the x0(t-1), x1(t-1), ..., x0(t-N), ..., xN(t-N) entries
                    replacement = f'x{idx%n_X_vars}(t-{idx//n_X_vars})'
                if idx >= n_X_vars: # Entries < n_X_vars are the actual x0, x1, ..., xN entries. Thus, they already have the correct ID
                    label_names[elem_idx] = elem.replace(f'x{idx}', replacement)

    # Scale data (mean = 0, stdev = 1)
    if X.shape[0] > 0 and X.shape[1] > 0: # StandardScaler requires at least one feature and one sample to work
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
        LCEN_model = None
        LCEN_params = np.empty(0)
        mse_train = np.var(y)
        mse_test = np.var(y_test)
        yhat_train = np.zeros(y.shape) # Ok to be 0 because y is scaled --> mean(y) is approximately 0
        yhat_test = np.zeros(y_test.shape)
    else:
        LCEN_model, LCEN_params, mse_train, mse_test, yhat_train, yhat_test = EN_fitting(X, y, X_test, y_test, alpha, l1_ratio)
    # Calculating information criteria values
    num_train = X.shape[0] # Typically written as n
    num_parameter = (LCEN_params!=0).flatten().sum() # Typically written as k
    AIC = num_train*np.log(mse_train) + 2*num_parameter # num_train * log(MSE) is one of the formulae to replace L. Shown in https://doi.org/10.1002/wics.1460, for example.
    AICc = AIC + 2*num_parameter*(num_parameter+1) / (num_train - num_parameter - 1)
    BIC = num_train*np.log(mse_train) + num_parameter*np.log(num_train)
    return (LCEN_model, LCEN_params, mse_train, mse_test, yhat_train, yhat_test, label_names, (AIC, AICc, BIC))

def forest_fitting(X, y, X_test, y_test, n_estimators = 100, max_depth = 10, min_samples_leaf = 0.1, max_features = 1.0, learning_rate = None, random_state = 0):
    """
    Fits data using a random forest or gradient-boosted decision trees

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
        Setting it to None, 'auto', or 1.0 uses all available features (= X.shape[1] = m).
    learning_rate : float or None, optional, default = None
        The learning rate for gradient-boosted decision trees.
        If None, a random forest regressor is used.
    random_state : int or None, optional, default = 0
        Seed for the random number generator.
    """
    if learning_rate is None or learning_rate == 0:
        forest = RandomForestRegressor(n_estimators, max_depth = max_depth, random_state = random_state, max_features = max_features, min_samples_leaf = min_samples_leaf)
    else:
        forest = GradientBoostingRegressor(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, random_state = random_state,
                                           max_features = max_features, min_samples_leaf = min_samples_leaf)
    forest.fit(X, y.flatten())
    # Predictions and MSEs
    yhat_train = forest.predict(X)
    yhat_test = forest.predict(X_test)
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return (forest, mse_train, mse_test, yhat_train, yhat_test)

def SVM_fitting(X, y, X_test, y_test, gamma = 'scale', C = 100, epsilon = 0.1, tol = 1e-4, max_iter = 10000):
    """
    Support Vector Machine-based regression

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    gamma : (float >=0) or str in {'scale', 'auto'}, optional, default = 'scale'
        Kernel coefficient for the 'rbf', 'poly', and 'sigmoid' kernels
        If 'scale', gamma = 1 / (X.shape[1] * X.var())
        If 'auto', gamma = 1 / X.shape[1]
    C : float > 0, optional, default = 100
        Parameter inversely proportional to the strength of the regularization
    epsilon : float >= 0, optional, default = 0.1
        Epsilon-tube within which no penalty is associated in the training loss function
    max_iter : int, optional, default = 10000
        The maximum number of iterations
    """
    SVM_model = SVR(gamma = gamma, C = C, epsilon = epsilon, tol = tol, max_iter = max_iter)
    SVM_model.fit(X, y.flatten())
    # Predictions and MSEs
    yhat_train = SVM_model.predict(X)
    yhat_test = SVM_model.predict(X_test)
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return (SVM_model, mse_train, mse_test, yhat_train, yhat_test)

def AdaBoost_fitting(X, y, X_test, y_test, n_estimators = 50, learning_rate = 0.1, random_state = 0):
    """
    Adaptive Boosting regression

    Parameters
    ----------
    X, y : Numpy array with shape N x m, N x 1
        Training data predictors and response.
    X_test, y_test : Numpy array with shape N_test x m, N_test x 1
        Testing data predictors and response.
    n_estimators : int, optional, default = 50
        The maximum number of estimators used.
        In case of perfect fit, the procedure is terminated early.
    learning_rate : float, optional, default = 0.1
        The learning rate applied at each boosting iteraction.
        If None, a random forest regressor is used.
    random_state : int or None, optional, default = 0
        Seed for the random number generator.
    """
    AdaB_model = AdaBoostRegressor(n_estimators = n_estimators, learning_rate = learning_rate, loss = 'square', random_state = random_state)
    AdaB_model.fit(X, y.flatten())
    # Predictions and MSEs
    yhat_train = AdaB_model.predict(X)
    yhat_test = AdaB_model.predict(X_test)
    mse_train = MSE(y, yhat_train)
    mse_test = MSE(y_test, yhat_test)

    return (AdaB_model, mse_train, mse_test, yhat_train, yhat_test)

def _feature_trans(X, X_test = None, degree = 2, interaction = True, trans_type = 'all', all_pos_X = True, all_pos_y = True):
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
    all_pos_X, all_pos_y : bool, optional, default = True
        Whether all entries in the data are positive or also contain some negative entries. This is done to avoid taking ...
            sqrt(k) for some k<0.
        This used to be determined within this function, but it leads to array size problems when some folds have negative ...
            entries and some do not.
    """
    X_test_out = None # Declaring this variable to avoid errors when doing the return statement

    # Polynomial transforms (and interaction terms)
    poly = PolynomialFeatures(degree, include_bias = False, interaction_only = trans_type.casefold() == 'simple_interaction', order = 'F') # Switching to Fortran order to speed up Elastic Net (and LCEN) [but does not make any significant difference]
    X_out = poly.fit_transform(X)
    label_names = poly.get_feature_names_out()
    interaction_column = np.array([' ' in elem for elem in label_names], dtype = bool) # To filter out interactions if user asked for a polynomial-only transform. Also for the log, sqrt, and inv terms below when poly_trans_only == False
    number_y_entries = X.shape[1] - len(all_pos_X) # Additional y entries within the array X = total entries - X entries. Is != 0 only when LCEN_transform_y == True
    all_pos = np.concatenate(( all_pos_X, np.tile(all_pos_y, number_y_entries) ))
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
        # if X_test is not None: # Columns where all entries are >= 0 for sqrt
        #     all_pos = np.all(X >= 0, axis = 0) & np.all(X_test >= 0, axis = 0)
        # else:
        #     all_pos = np.all(X >= 0, axis = 0)
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
        Xlog_trans = poly.fit_transform(Xlog)[:, ~interaction_column]
        X_out = np.column_stack((X_out, Xlog_trans))
        temp_label_names = poly.get_feature_names_out()[~interaction_column]
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
            Xlog_test_trans = poly.fit_transform(Xlog_t)[:, ~interaction_column]
            X_test_out = np.column_stack((X_test_out, Xlog_test_trans))
        # sqrt transform
        X_out = np.column_stack((X_out, Xsqrt))
        temp_label_names = [f'sqrt(x{number})' for number in range(X.shape[1]) if all_pos[number]]
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            X_test_out = np.column_stack((X_test_out, Xsqrt_t))
        # Inverse transform
        if not interaction:
            Xinv_trans = poly.fit_transform(Xinv)[:, ~interaction_column]
            temp_label_names = poly.get_feature_names_out()[~interaction_column]
        else:
            Xinv_trans = poly.fit_transform(Xinv)
            temp_label_names = poly.get_feature_names_out()
        X_out = np.column_stack((X_out, Xinv_trans))
        for idx in range(len(temp_label_names)): # Converting the names from x0-like to ln(x0)-like
            temp_label_names[idx] = temp_label_names[idx].translate({ord(i): '*' for i in ' '}) # str.translate replaces the characters on the right of the "for i in" (in this case, a whitespace) with an asterisk
            temp_label_names[idx] = f'1/({temp_label_names[idx]})' # 1/(x1^3) is the same as (1/x1)^3, so we do not need the fancy manipulations used above in the ln transform naming
        label_names = np.concatenate((label_names, temp_label_names))
        if X_test is not None:
            if not interaction:
                Xinv_test_trans = poly.fit_transform(Xinv_t)[:, ~interaction_column]
            else:
                Xinv_test_trans = poly.fit_transform(Xinv_t)
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
