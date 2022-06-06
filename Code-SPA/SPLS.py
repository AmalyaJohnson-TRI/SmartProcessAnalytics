"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
Modified by Pedro Seber
"""
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

# Loading the R libraries
utils = importr("utils")
d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
spls = importr('spls')


def mse(y, yhat):
    """
    This function calculates the MSE between the real values y and predictions yhat
    Input: y: N x 1 array
           yhat: N x 1 array
           
    Output: mse float
    """
    return np.sum((yhat-y)**2)/y.shape[0]


def SPLS_fitting_method(X, y, X_test, y_test, K = None, eta = None, v = 5, eps = 1e-4, maxstep = 1000):
    """
    This function uses rpy2 to call SPLS, an R package
    https://cran.r-project.org/web/packages/spls/vignettes/spls-example.pdf
    "The responses and the predictors are assumed to be numerical and should not contain missing values.
    As part of pre-processing, the predictors are centered and scaled and the responses are centered
    automatically as default by the package ‘spls’"

    Input:
        X: independent variables of size N x m
        y: dependent variable of size N x 1
        X_test: independent variables of size N_test x m
        y_test: dependent variable of size N_test x 1
        v: int, v fold cross-validation, default = 5 using the default CV in SPLS package
        K: int, the number of latent variable, ranging from 1 to min(m, (v-1)N/v), default using cross validation to determin
        eta: float, sparsity tuning parameter: ranging from 0 to 1, default seq(0,1,0.05), default, using cross validation to detemine
        
    Output:
        tuple (trained_model, model_params, mse_train, mse_test, yhat_train, yhat_test)
        trained_model: spls model type
        model_params: np_array m x 1
        """

    # Data preparation
    rpy2.robjects.numpy2ri.activate()
    
    # Convert training data numpy to R vector/matrix
    nr,nc = X.shape
    Xr = ro.r.matrix(X, nrow=nr, ncol=nc)
    ro.r.assign("X", Xr)
    nry,ncy = y.shape
    yr = ro.r.matrix(y,nrow=nry,ncol=ncy)
    ro.r.assign("y", yr)
    
    # CV fitting to choose K and eta if not given
    if K is None and eta is None:
        m = nc
        N = nr
        f = spls.cv_spls(Xr, yr, K=ro.r.seq(1,min(m,int((v-1)/v*N)),1), eta=ro.r.seq(0, 0.95, 0.05), fold=5, plot_it = False, scale_x=False, scale_y=False, eps=eps)
        eta_opt = float(np.array(f.rx2('eta.opt')))
        K_opt = int(np.array(f.rx2('K.opt')))
    elif K is None and eta is not None:
        m = nc
        N = nr
        f = spls.cv_spls(Xr, yr, K=ro.r.seq(1,min(m,int((v-1)/v*N)),1), eta=eta, fold=5, plot_it = False, scale_x=False, scale_y=False, eps=eps)
        eta_opt = float(np.array(f.rx2('eta.opt')))
        K_opt = int(np.array(f.rx2('K.opt')))
    elif K is not None and eta is None:
        f = spls.cv_spls(Xr, yr, K=K, eta=ro.r.seq(0, 1, 0.05), fold=5, plot_it = False, scale_x=False, scale_y=False)
        eta_opt = float(np.array(f.rx2('eta.opt')))
        K_opt = int(np.array(f.rx2('K.opt')))
    else:
        # K and eta were specified by the user
        K_opt = K
        eta_opt = eta
    
    # Fit the final model
    SPLS_model = spls.spls(Xr, yr, eta = eta_opt, K = K_opt, scale_x=False, scale_y=False, eps=eps, maxstep = maxstep)

    # Extract coefficients
    SPLS_params = spls.predict_spls(SPLS_model, type = "coefficient")
    SPLS_params = np.array(SPLS_params)

    # Predictions from training and testing data
    yhat_train = np.dot(X, SPLS_params)
    yhat_test = np.dot(X_test, SPLS_params)

    # MSE calculations
    mse_train = mse(y, yhat_train)
    mse_test = mse(y_test, yhat_test)

    return (SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test)

