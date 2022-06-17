"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com, https://github.com/vickysun5/SmartProcessAnalytics
Modified by Pedro Seber, https://github.com/PedroSeber/SmartProcessAnalytics
"""
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, TimeSeriesSplit, GroupKFold, GroupShuffleSplit
import regression_models as rm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import nonlinear_regression as nr
from sklearn.model_selection import train_test_split
import nonlinear_regression_other as nro
from sklearn.feature_selection import VarianceThreshold

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
    if Type == 'MC':
        CV = ShuffleSplit(n_splits = Nr, test_size = 1/K, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'Single':
        X, X_test, y, y_test = train_test_split(X, y, test_size = 1/K, random_state = random_state)
        yield (X, y, X_test, y_test)
    elif Type == 'KFold':
        CV = KFold(n_splits = int(K), random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'Re_KFold':
        CV = RepeatedKFold(n_splits = int(K), n_repeats = Nr, random_state = random_state)
        for train_index, val_index in CV.split(X, y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
    elif Type == 'Timeseries':
        TS = TimeSeriesSplit(n_splits = int(K))
        for train_index, val_index in TS.split(X):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])    
    elif Type == 'Single_group':
        label = np.unique(group)
        num = int(len(label)/K)
        final_list = np.squeeze(group == label[0])
        for i in range(1, num):
            final_list = final_list | np.squeeze(group == label[i])
        yield(X[~final_list], y[~final_list], X[final_list], y[final_list])
    elif Type == 'Group':
        label = np.unique(group)
        for i in range(len(label)):
            yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'Group_no_extrapolation':
        label = np.unique(group)
        for i in range(len(label)):
            if min(label) < label[i] and label[i] < max(label):
                yield(X[np.squeeze(group != label[i])], y[np.squeeze(group != label[i])], X[np.squeeze(group == label[i])], y[np.squeeze(group == label[i])])
    elif Type == 'GroupKFold':
        gkf = GroupKFold(n_splits = int(K))
        for train_index, val_index in gkf.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])    
    elif Type == 'GroupShuffleSplit':
        gss = GroupShuffleSplit(n_splits = int(Nr), test_size = 1 / K, random_state = random_state)
        for train_index, val_index in gss.split(X, y, groups = group):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])    
    elif Type == 'No_CV':
        yield (X, y, X, y)
    elif Type == 'Single_ordered':
        num = X.shape[0]
        yield (X[:num-round(X.shape[0]*1/K):], y[:num-round(X.shape[0]*1/K):], X[num-round(X.shape[0]*1/K):], y[num-round(X.shape[0]*1/K):])        
    else:
        print('Wrong type specified for data partition')

def CV_mse(model_name, X, y, X_test, y_test, cv_type = 'Re_KFold', K_fold = 5, Nr = 1000, eps = 1e-4, alpha_num = 20, group = None, round_number = '', **kwargs):
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
    cv_type : str, optional, default = None
        Which cross validation method to use.
    K_fold : int, optional, default = 5
        Number of folds used in cross validation.
    Nr : int, optional, default = 10
        Number of CV repetitions used when cv_type in {'MC', 'Re_KFold', 'GroupShuffleSplit'}.
    eps : float, optional, default = 1e-4
        Tolerance. TODO: expand on this.
    alpha_num : int, optional, default = 20
        Penalty weight used when model_name in {'RR', 'EN', 'ALVEN', 'DALVEN', 'DALVEN_full_nonlinear'}.
    **kwargs : dict, optional
        Non-default hyperparameters for model fitting.
    """
    if model_name == 'EN':
        EN = rm.model_getter(model_name)
        

        if 'l1_ratio' not in kwargs:
            #kwargs['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50]
            kwargs['l1_ratio'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
            
        MSE_result = np.zeros((alpha_num,len(kwargs['l1_ratio'])))

        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
            for j in range(len(kwargs['l1_ratio'])):
                if kwargs['l1_ratio'][j] == 0:
                    alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/0.0001
                    kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]
                    for i in range(alpha_num):
                        clf = Ridge(alpha=kwargs['alpha'][i],fit_intercept=False).fit(X_train, y_train)
                        mse = np.sum((clf.predict(X_val)-y_val)**2)/y_val.shape[0]                                                
                        MSE_result[i,j] += mse

                else:
                    alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/kwargs['l1_ratio'][j]
                    kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), alpha_num)[::-1]
                    for i in range(alpha_num):
                        _, _, _, mse, _, _ = EN(X_train, y_train, X_val, y_val, alpha = kwargs['alpha'][i], l1_ratio = kwargs['l1_ratio'][j])
                        MSE_result[i,j] += mse

        MSE_result = MSE_result/counter
                
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        l1_ratio = kwargs['l1_ratio'][ind[1]]
        
        if l1_ratio != 0:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/l1_ratio
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), alpha_num)[::-1]
           
            alpha = kwargs['alpha'][ind[0]]
        else:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/0.0001
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]
           
            alpha = kwargs['alpha'][ind[0]]
            
            
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        
        #fit the final model using opt hyper_params
        if l1_ratio == 0:
            EN_model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
            EN_params= EN_model.coef_.reshape(-1,1)
            yhat_train = EN_model.predict(X)
            yhat_test = EN_model.predict(X_test)
            mse_train = np.sum((yhat_train-y)**2)/y.shape[0]  
            mse_test = np.sum((yhat_test-y_test)**2)/y_test.shape[0]  
        else:
            EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test = EN(X, y, X_test, y_test, alpha = alpha, l1_ratio = l1_ratio)
        
        return(hyper_params, EN_model, EN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
    ########################################################################################################################################   
        
    elif model_name == 'SPLS':
        SPLS = rm.model_getter(model_name)
        
        if cv_type != 'Group' and cv_type != 'Group_no_extrapolation' and cv_type != 'GroupKFold' and cv_type != 'GroupShuffleSplit':
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)),min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)))
            if 'eta' not in kwargs:
                kwargs['eta'] = np.linspace(0,1,20, endpoint = False)[::-1] #eta = 0 use normal PLS
           
            MSE_result = np.zeros((len(kwargs['K']),len(kwargs['eta'])))
            
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
                counter += 1
                for i in range(len(kwargs['K'])):
                    for j in range(len(kwargs['eta'])):
                        _, _, _, mse, _, _ = SPLS(X_train, y_train, X_val, y_val, K = int(kwargs['K'][i]), eta = kwargs['eta'][j], eps = eps)
                        MSE_result[i,j] += mse
        
        else:
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int(X.shape[0]-1)),min(X.shape[1],int(X.shape[0]-1)))
            if 'eta' not in kwargs:
                kwargs['eta'] = np.linspace(0,1,20, endpoint = False)[::-1] #eta = 0 use normal PLS
           
            MSE_result = np.zeros((len(kwargs['K']),len(kwargs['eta'])))
            
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
                counter += 1
#                print(counter)
#                start_time = timeit.default_timer()
                for i in range(len(kwargs['K'])):
                    for j in range(len(kwargs['eta'])):
                        if kwargs['K'][i] > X_train.shape[0]-1:
                            mse = 10000
                        ######should be remove in the future##########################
                        elif kwargs['K'][i] > 30:
                            mse = 10000
                        else:
                            _, _, _, mse, _, _ = SPLS(X_train, y_train, X_val, y_val, K = int(kwargs['K'][i]), eta = kwargs['eta'][j], eps = eps)
                        
                        MSE_result[i,j] += mse
#                print(timeit.default_timer() - start_time)   
                    
        MSE_result = MSE_result/counter
            
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        K = kwargs['K'][ind[0]]
        eta = kwargs['eta'][ind[1]]
            
        hyper_params = {}
        hyper_params['K'] = int(K)
        hyper_params['eta'] = eta
                
        
        #fit the final model using opt hyper_params
        SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test = SPLS(X, y, X_test, y_test, eta = eta, K = K)
        
        return(hyper_params, SPLS_model, SPLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
    
    ########################################################################################################################################   
    
    elif model_name == 'LASSO':
        LASSO = rm.model_getter(model_name)
        
        if 'alpha' not in kwargs:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), alpha_num)[::-1]

            #[1e-5, 5*1e-5, 1e-4, 5*1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 50]
        
        MSE_result = np.zeros(len(kwargs['alpha']))
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter +=1
            for i in range(len(kwargs['alpha'])):
                _, _, _, mse, _, _ = LASSO(X_train, y_train, X_val, y_val, alpha = kwargs['alpha'][i])
                MSE_result[i] += mse
                    
        MSE_result = MSE_result/counter
               
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        alpha = kwargs['alpha'][ind]
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        
        #fit the final model using opt hyper_params
        LASSO_model, LASSO_params, mse_train, mse_test, yhat_train, yhat_test = LASSO(X, y, X_test, y_test, alpha = alpha)
        
        return(hyper_params, LASSO_model, LASSO_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])

    ########################################################################################################################################
    elif model_name == 'POLY':
        #Nonlinear feature transformation through poly nomials
        OLS = rm.model_getter('OLS')


        if 'degree' not in kwargs:
            kwargs['degree'] = [2,3,4]
        
        if 'interaction' not in kwargs:
            kwargs['interaction'] = True
        if 'power' not in kwargs:
            kwargs['power'] = True
        
               
           
        MSE_result = np.zeros(len(kwargs['degree']))
       
        for d in range(len(kwargs['degree'])):
            X_trans, _ = nr.poly_feature(X, X_test, degree = kwargs['degree'][d], interaction = kwargs['interaction'], power = kwargs['power'])
            X_trans=np.hstack((np.ones([X_trans.shape[0],1]),X_trans))
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X_trans, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):                
                counter += 1
                _, _, _, mse, _, _ = OLS(X_train, y_train, X_val, y_val)
               
                MSE_result[d] += mse
                    
        MSE_result = MSE_result/counter
        
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        degree = kwargs['degree'][ind[0]]
        
        hyper_params = {}
        hyper_params['degree'] = int(degree)

        #fit the final model using opt hyper_params
        X_trans, X_trans_test = nr.poly_feature(X, X_test, degree = degree, interaction = kwargs['interaction'], power = kwargs['power'])
        X_trans=np.hstack((np.ones([X_trans.shape[0],1]),X_trans))
        X_trans_test=np.hstack((np.ones([X_trans_test.shape[0],1]),X_trans_test))

        POLY_model, POLY_params, mse_train, mse_test, yhat_train, yhat_test = OLS(X_trans, y, X_trans_test, y_test)
        
        return(hyper_params, POLY_model, POLY_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
     ########################################################################################################################################
         
    elif model_name == 'PLS':
        
        if cv_type != 'Group' and cv_type != 'Group_no_extrapolation' and cv_type != 'GroupKFold' and cv_type != 'GroupShuffleSplit':
          
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)),min(X.shape[1],int((K_fold-1)/K_fold*X.shape[0]-1)))
           
            MSE_result = np.zeros((len(kwargs['K'])))

            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group=group):
                counter += 1
                for i in range(len(kwargs['K'])):
#                    print(str(kwargs['K'][i])+'  '+str(X_train.shape[0]))
                    PLS = PLSRegression(scale = False, n_components=int(kwargs['K'][i]), tol = eps).fit(X_train,y_train)
                    PLS_para = PLS.coef_.reshape(-1,1)
                    yhat = np.dot(X_val, PLS_para)
                    MSE_result[i] += rm.mse(y_val, yhat)
        
        else:
            if 'K' not in kwargs:
                kwargs['K'] = np.linspace(1, min(X.shape[1],int(X.shape[0]-1)),min(X.shape[1],int(X.shape[0]-1)))
                     
            MSE_result = np.zeros((len(kwargs['K'])))
            
            counter = 0
            for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group=group):
                counter += 1
                for i in range(len(kwargs['K'])):
                    if kwargs['K'][i] > X_train.shape[0]-1:
                        MSE_result[i] = 10000
                    else:
                        PLS = PLSRegression(scale = False, n_components=int(kwargs['K'][i]), tol = eps).fit(X_train,y_train)
                        PLS_para = PLS.coef_.reshape(-1,1)
                        yhat = np.dot(X_val, PLS_para)
                        MSE_result[i] += rm.mse(y_val, yhat)
                              
                    
        MSE_result = MSE_result/counter
            
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape) 
        K = kwargs['K'][ind[0]]
            
        hyper_params = {}
        hyper_params['K'] = int(K)
                
        
        #fit the final model using opt hyper_params
        PLS_model = PLSRegression(scale = False, n_components=int(hyper_params['K'])).fit(X,y)
        PLS_params = PLS_model.coef_.reshape(-1,1)
        
        yhat_train = np.dot(X, PLS_params)
        yhat_test = np.dot(X_test, PLS_params)
        
        mse_train = rm.mse(yhat_train,y)
        mse_test = rm.mse(yhat_test,y_test)
        
        return(hyper_params, PLS_model, PLS_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
     


     ########################################################################################################################################
         
    elif model_name == 'RR':
        
        if 'alpha' not in kwargs:
            alpha_max = (np.sqrt(np.sum(np.dot(X.T,y) ** 2, axis=1)).max())/X.shape[0]/0.0001
            kwargs['alpha'] = np.logspace(np.log10(alpha_max * eps/100), np.log10(alpha_max), alpha_num)[::-1]
   
        MSE_result = np.zeros((len(kwargs['alpha'])))
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter+=1
            for i in range(len(kwargs['alpha'])):
                RR = Ridge(alpha = kwargs['alpha'][i], fit_intercept = False).fit(X_train, y_train)
                Para = RR.coef_.reshape(-1,1)
                yhat = np.dot(X_val, Para)
                MSE_result[i] += rm.mse(y_val, yhat)
                                        
        MSE_result = MSE_result/counter
            
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        alpha = kwargs['alpha'][ind[0]]
            
        hyper_params = {}
        hyper_params['alpha'] = alpha
                
        
        #fit the final model using opt hyper_params
        RR_model = Ridge(alpha = hyper_params['alpha'], fit_intercept = False).fit(X,y)
        RR_params = RR_model.coef_.reshape(-1,1)
        
        yhat_train = np.dot(X, RR_params)
        yhat_test = np.dot(X_test, RR_params)
        
        mse_train = rm.mse(yhat_train,y)
        mse_test = rm.mse(yhat_test,y_test)
        
        return(hyper_params, RR_model, RR_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
 
    ########################################################################################################################################   
    
    elif model_name == 'ALVEN':
        ALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2,3]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['ALVEN_select_pvalue'] = 0.10
            
            
        MSE_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio'])))
    
        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
#            print('*****'+str(counter))
            for k in range(len(kwargs['degree'])):
                for j in range(len(kwargs['l1_ratio'])):
                    for i in range(alpha_num):
                        _, _, _, mse, _, _ , _, _= ALVEN(X_train, y_train, X_val, y_val, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['ALVEN_select_pvalue'], trans_type = kwargs['trans_type'])
                        MSE_result[k,i,j] += mse

        MSE_result = MSE_result/counter


            
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        degree = kwargs['degree'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[2]]
       
        ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index= ALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree =  degree, tol = eps , alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['ALVEN_select_pvalue'], trans_type = kwargs['trans_type'])
  
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        hyper_params['degree'] = degree
        hyper_params['retain_index'] = retain_index
        
        #get the name for the retained
        if kwargs['trans_type'] == 'auto':
            Xtrans,_ = nr.feature_trans(X, degree = degree, interaction = 'later')
        else:
            Xtrans, _ = nr.poly_feature(X, degree = degree, interaction = True, power = True)

        sel = VarianceThreshold(threshold=eps).fit(Xtrans)



        if kwargs['label_name'] :
            if 'xticks' in kwargs:
                list_name = kwargs['xticks']
            else:
                list_name =['x'+str(i) for i in range(1,np.shape(X)[1]+1)]
            
            if kwargs['trans_type'] == 'auto':
                if degree == 1:
                    list_name_final = list_name + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]
                
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+[name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                      [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name]
                    
                if degree == 3:
                    list_name_final = list_name[:]
                    
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                            
                    for i in range(X.shape[1]-2):
                        for j in range(i+1,X.shape[1]-1):
                            for k in range(j+1,X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+\
                                       [name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                       [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name] +\
                                       [name +'^3' for name in list_name]+['(log'+ name + ')^3' for name in list_name]  + ['1/' +name+'^3' for name in list_name]+\
                                       [name +'^2.5' for name in list_name] +['(log' +name +')^2/' + name for name in list_name]+ ['log(' +name +')/sqrt(' + name +')' for name in list_name]+ ['log(' +name +')/' + name +'^2' for name in list_name]+\
                                       [name +'^-1.5' for name in list_name]
            else:
                if degree == 1:
                    list_name_final = list_name
                
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                
                if degree == 3:
                    list_name_final = list_name[:]

                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            for k in range(j, X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                                
                        
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final= [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []


        return(hyper_params, ALVEN_model, ALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind], list_name_final)
     ########################################################################################################################################   
         
    elif model_name == 'RF':
        
        if 'max_depth' not in kwargs:
            kwargs['max_depth'] = [2,3,5,10,15,20,40]
        if 'n_estimators' not in kwargs:
            kwargs['n_estimators'] = [10, 50, 100, 200]
        if 'min_samples_leaf' not in kwargs:
            kwargs['min_samples_leaf'] = [0.0001]#0.02,0.05, 0.1] #, 0.05 ,0.1, 0.2] # 0.3, 0.4]
        
        MSE_result = np.zeros((len(kwargs['max_depth']),len(kwargs['n_estimators']),len(kwargs['min_samples_leaf'])))
        
        count = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            count += 1
#            print(count)
            for i in range(len(kwargs['max_depth'])):
                for j in range(len(kwargs['n_estimators'])):
                    for k in range(len(kwargs['min_samples_leaf'])):
                        _, _, mse, _,_ = nro.RF_fitting(X_train, y_train, X_val,y_val, n_estimators = kwargs['n_estimators'][j], max_depth = kwargs['max_depth'][i], min_samples_leaf=kwargs['min_samples_leaf'][k])
                        MSE_result[i,j,k] += mse
                                        
        MSE_result = MSE_result/count
            
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        max_depth = kwargs['max_depth'][ind[0]]
        n_estimators = kwargs['n_estimators'][ind[1]]
        min_samples_leaf = kwargs['min_samples_leaf'][ind[2]]
            
        hyper_params = {}
        hyper_params['max_depth'] = max_depth
        hyper_params['n_estimators'] = n_estimators
        hyper_params['min_samples_leaf'] = min_samples_leaf
        
        #fit the final model using opt hyper_params
        RF_model, mse_train, mse_test, yhat_train, yhat_test = nro.RF_fitting(X, y, X_test, y_test, n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf)
        
        return(hyper_params, RF_model, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
 
    ########################################################################################################################################   
  
    elif model_name == 'SVR':
        
        if 'C' not in kwargs:
            kwargs['C'] = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        if 'gamma' not in kwargs:
            gd = 1/X.shape[1]
            kwargs['gamma'] = [gd/50, gd/10, gd/5, gd/2, gd, gd*2, gd*5, gd*10, gd*50]
        if 'epsilon' not in kwargs:
            kwargs['epsilon'] = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3]
        
        MSE_result = np.zeros((len(kwargs['C']),len(kwargs['gamma']),len(kwargs['epsilon'])))

        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
#            print(counter)
            for i in range(len(kwargs['C'])):
                for j in range(len(kwargs['gamma'])):
                    for k in range(len(kwargs['epsilon'])):
                        _, _, mse, _,_ = nro.SVR_fitting(X_train, y_train, X_val,y_val,
                                                     C=kwargs['C'][i], gamma=kwargs['gamma'][j], epsilon=kwargs['epsilon'][k])
                        MSE_result[i,j,k] += mse
                                        
        MSE_result = MSE_result/counter
            
        #find the min value, if there is a tie, only the first occurence is returned
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        C = kwargs['C'][ind[0]]
        gamma = kwargs['gamma'][ind[1]]
        epsilon = kwargs['epsilon'][ind[2]]

        hyper_params = {}
        hyper_params['C'] = C
        hyper_params['gamma'] = gamma
        hyper_params['epsilon'] = epsilon
        
        #fit the final model using opt hyper_params
        SVR_model, mse_train, mse_test, yhat_train, yhat_test =  nro.SVR_fitting(X, y, X_test,y_test,
                                                                                C=C, gamma=gamma, epsilon=epsilon)
        
        return(hyper_params, SVR_model, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind])
 
    ########################################################################################################################################   
      
    elif model_name == 'DALVEN':
        DALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2,3]
            
        if 'lag' not in kwargs:
            kwargs['lag'] = [i+1 for i in range(40)]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
            
            
        MSE_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag'])))
    
        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
#            print('*****'+str(counter))
            for k in range(len(kwargs['degree'])):
                for j in range(len(kwargs['l1_ratio'])):
                    for i in range(alpha_num):
                        for t in range(len(kwargs['lag'])):
                            _, _, _, mse, _, _ , _, _,_= DALVEN(X_train, y_train, X_val, y_val, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                            MSE_result[k,i,j,t] += mse

        MSE_result = MSE_result/counter


            
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        degree = kwargs['degree'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[2]]
        lag = kwargs['lag'][ind[3]]
       
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,_= DALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree =  degree, lag = lag, tol = eps , alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
  
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        hyper_params['degree'] = degree
        hyper_params['lag'] = lag
        hyper_params['retain_index'] = retain_index

        #get the name for the retained
        if kwargs['trans_type'] == 'auto':
            Xtrans,_ = nr.feature_trans(X, degree = degree, interaction = 'later')
        else:
            Xtrans, _ = nr.poly_feature(X, degree = degree, interaction = True, power = True)


    
        #lag padding for X
        XD = Xtrans[lag:]
        for i in range(lag):
            XD = np.hstack((XD,Xtrans[lag-1-i:-i-1]))
            
        #lag padding for y in design matrix
        for i in range(lag):
            XD = np.hstack((XD,y[lag-1-i:-i-1]))
        
        #remove feature with 0 variance
        sel = VarianceThreshold(threshold=eps).fit(XD)



        if kwargs['label_name'] :
            if 'xticks' in kwargs:
                list_name = kwargs['xticks']
            else:
                list_name =['x'+str(i) for i in range(1,np.shape(X)[1]+1)]
            
            if kwargs['trans_type'] == 'auto':
                if degree == 1:
                    list_name_final = list_name + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                    
                      
                        
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+[name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                      [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name]
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        


                    
                if degree == 3:
                    list_name_final = list_name[:]
                    
                    for i in range(X.shape[1]-1):
                        for j in range(i+1,X.shape[1]):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                            
                    for i in range(X.shape[1]-2):
                        for j in range(i+1,X.shape[1]-1):
                            for k in range(j+1,X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+\
                                       [name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                       [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name] +\
                                       [name +'^3' for name in list_name]+['(log'+ name + ')^3' for name in list_name]  + ['1/' +name+'^3' for name in list_name]+\
                                       [name +'^2.5' for name in list_name] +['(log' +name +')^2/' + name for name in list_name]+ ['log(' +name +')/sqrt(' + name +')' for name in list_name]+ ['log(' +name +')/' + name +'^2' for name in list_name]+\
                                       [name +'^-1.5' for name in list_name]

                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        




            else:
                if degree == 1:
                    list_name_final = list_name
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        

                
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        

                
                if degree == 3:
                    list_name_final = list_name[:]

                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            for k in range(j, X.shape[1]):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                  
                    list_copy = list_name_final[:]
                    
                    for i in range(lag):
                        list_name_final = list_name_final + [s + '(t-' + str(i+1) + ')' for s in list_copy]
                    for i in range(lag):
                        list_name_final = list_name_final + ['y(t-' + str(i+1) +')' ] 
                        

                                
                        
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final = [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []


        return(hyper_params, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind], list_name_final)
     



    ########################################################################################################################################   
      
    elif model_name == 'DALVEN_full_nonlinear':
        DALVEN = rm.model_getter(model_name)
        
        
        if 'l1_ratio' not in kwargs:
            kwargs['l1_ratio'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99][::-1]
        if 'degree' not in kwargs:
            kwargs['degree'] = [1,2] #,3]
            
        if 'lag' not in kwargs:
            kwargs['lag'] = [i+1 for i in range(40)]
        
        if 'label_name' not in kwargs:
            kwargs['label_name'] = False
        
        if 'trans_type' not in kwargs:
            kwargs['trans_type'] = 'auto'
        
        if 'select_value' not in kwargs:
            kwargs['select_pvalue'] = 0.05
            
            
        MSE_result = np.zeros((len(kwargs['degree']),alpha_num,len(kwargs['l1_ratio']), len(kwargs['lag'])))
    
        #check if the data is zscored, score back:
        #########################to be continue###################################
        
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr, group = group):
            counter += 1
#            print('*****'+str(counter))
            for k in range(len(kwargs['degree'])):
                for j in range(len(kwargs['l1_ratio'])):
                    for i in range(alpha_num):
                        for t in range(len(kwargs['lag'])):
                            _, _, _, mse, _, _ , _, _,_= DALVEN(X_train, y_train, X_val, y_val, alpha = i, l1_ratio = kwargs['l1_ratio'][j],
                                                      degree = kwargs['degree'][k], lag = kwargs['lag'][t], tol = eps , alpha_num = alpha_num, cv = True,
                                                      selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
                            MSE_result[k,i,j,t] += mse

        MSE_result = MSE_result/counter


            
        #find the min value, if there is a tie, only the first occurence is returned, and fit the final model
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        degree = kwargs['degree'][ind[0]]
        l1_ratio = kwargs['l1_ratio'][ind[2]]
        lag = kwargs['lag'][ind[3]]
       
        DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, alpha, retain_index,_= DALVEN(X,y, X_test, y_test, alpha = ind[1], l1_ratio = l1_ratio,
                                                                                           degree =  degree, lag = lag, tol = eps , alpha_num = alpha_num, cv = False,
                                                                                           selection = 'p_value', select_value = kwargs['select_pvalue'], trans_type = kwargs['trans_type'])
  
        
        hyper_params = {}
        hyper_params['alpha'] = alpha
        hyper_params['l1_ratio'] = l1_ratio
        hyper_params['degree'] = degree
        hyper_params['lag'] = lag
        hyper_params['retain_index'] = retain_index



        #lag padding for X
        XD = X[lag:]
        for i in range(lag):
            XD = np.hstack((XD,X[lag-1-i:-i-1]))
            
        #lag padding for y in design matrix
        for i in range(lag):
            XD = np.hstack((XD,y[lag-1-i:-i-1]))
            
        #get the name for the retained
        if kwargs['trans_type'] == 'auto':
            XD,_ = nr.feature_trans(XD, degree = degree, interaction = 'later')
        else:
            XD, _ = nr.poly_feature(XD, degree = degree, interaction = True, power = True)

        
        #remove feature with 0 variance
        sel = VarianceThreshold(threshold=eps).fit(XD)



        if kwargs['label_name'] :
            list_name =['x'+str(i) for i in range(1,np.shape(X)[1]+1)]
            list_copy = list_name[:]

            for i in range(lag):
                list_name = list_name + [s + '(t-' + str(i+1) + ')' for s in list_copy]
            for i in range(lag):
                list_name = list_name + ['y(t-' + str(i+1) +')' ] 
                    
            
            if kwargs['trans_type'] == 'auto':
                if degree == 1:
                    list_name_final = list_name + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]
                 
                        
                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(len(list_name)-1):
                        for j in range(i+1,len(list_name)):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+[name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                      [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name]


                    
                if degree == 3:
                    list_name_final = list_name[:]
                    
                    for i in range(len(list_name)-1):
                        for j in range(i+1,len(list_name)):
                            list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]]
                            
                    for i in range(len(list_name)-2):
                        for j in range(i+1,len(list_name)-1):
                            for k in range(j+1,len(list_name)):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                    list_name_final  = list_name_final + ['log('+ name + ')' for name in list_name] + ['sqrt(' +name+')' for name in list_name]+['1/' +name for name in list_name]+\
                                       [name +'^2' for name in list_name]+['(log'+ name + ')^2' for name in list_name] + ['1/' +name+'^2' for name in list_name]+\
                                       [name +'^1.5' for name in list_name]+ ['log(' +name +')/' + name for name in list_name]+ ['1/' +name+'^0.5' for name in list_name] +\
                                       [name +'^3' for name in list_name]+['(log'+ name + ')^3' for name in list_name]  + ['1/' +name+'^3' for name in list_name]+\
                                       [name +'^2.5' for name in list_name] +['(log' +name +')^2/' + name for name in list_name]+ ['log(' +name +')/sqrt(' + name +')' for name in list_name]+ ['log(' +name +')/' + name +'^2' for name in list_name]+\
                                       [name +'^-1.5' for name in list_name]                    




            else:
                if degree == 1:
                    list_name_final = list_name
                    

                if degree == 2:
                    list_name_final = list_name[:]
                    for i in range(X.shape[1]):
                        for j in range(i, X.shape[1]):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
       

                
                if degree == 3:
                    list_name_final = list_name[:]

                    for i in range(len(list_name)):
                        for j in range(i, len(list_name)):
                            list_name_final = list_name_final +[list_name[i]+'*'+list_name[j]]
                    
                    for i in range(len(list_name)):
                        for j in range(i, len(list_name)):
                            for k in range(j, len(list_name)):
                                list_name_final = list_name_final + [list_name[i]+'*'+list_name[j]+'*'+list_name[k]]
                  

                                
                        
            index = list(sel.get_support())
            list_name_final = [x for x, y in zip(list_name_final, index) if y]
            list_name_final = [x for x, y in zip(list_name_final, retain_index) if y]
        
        else:
            list_name_final =  []


        return(hyper_params, DALVEN_model, DALVEN_params, mse_train, mse_test, yhat_train, yhat_test, MSE_result[ind], list_name_final)

    elif model_name == 'RNN':
        import timeseries_regression_RNN as RNN
        input_size_x = X.shape[1]

        # Model architecture
        if 'cell_type' not in kwargs:
            kwargs['cell_type'] = ['basic']
        if 'activation' not in kwargs:
            kwargs['activation'] = ['tanh']
        if 'RNN_layers' not in kwargs:
            kwargs['RNN_layers'] = [[input_size_x]]

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
        if 'val_ratio' not in kwargs:
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
                
        MSE_result = np.zeros( (len(kwargs['cell_type']), len(kwargs['activation']), len(kwargs['RNN_layers'])) )
        
        for counter, (X_train, y_train, X_val, y_val) in enumerate(CVpartition(X, y, cv_type, K_fold, Nr, group = group)):
            for i in range(len(kwargs['cell_type'])):
                for j in range(len(kwargs['activation'])):
                    for k in range(len(kwargs['RNN_layers'])):
                        _, _, _, _, _, val_loss, _ = RNN.timeseries_RNN_feedback_single_train(X, y, X_val, y_val, None, None, kwargs['val_ratio'], kwargs['cell_type'][i],
                                kwargs['activation'][j], kwargs['RNN_layers'][k], kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'],
                                kwargs['lambda_l2_reg'], kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test,
                                output_prob_test, state_prob_test, kwargs['max_checks_without_progress'], kwargs['epoch_before_val'], kwargs['save_location'], plot = False)
                            
                        MSE_result[i, j, k] += val_loss
                                
        MSE_result = MSE_result / (counter+1)
    
        # Min IC value (first occurrence)
        ind = np.unravel_index(np.argmin(MSE_result, axis=None), MSE_result.shape)
        cell_type = kwargs['cell_type'][ind[0]]
        activation = kwargs['activation'][ind[1]]
        RNN_layers = kwargs['RNN_layers'][ind[2]]
       
        prediction_train, prediction_val, prediction_test, _, train_loss_final, val_loss_final, test_loss_final = RNN.timeseries_RNN_feedback_single_train(X, y, None, None, X_test, y_test,
                kwargs['val_ratio'], cell_type, activation, RNN_layers, kwargs['batch_size'], kwargs['epoch_overlap'], kwargs['num_steps'], kwargs['learning_rate'], kwargs['lambda_l2_reg'],
                kwargs['num_epochs'], kwargs['input_prob'], kwargs['output_prob'], kwargs['state_prob'], input_prob_test, output_prob_test, state_prob_test, kwargs['max_checks_without_progress'],
                kwargs['epoch_before_val'], kwargs['save_location'], plot = kwargs['plot'])

        hyperparams = {}
        hyperparams['cell_type'] = cell_type
        hyperparams['activation'] = activation
        hyperparams['RNN_layers'] = RNN_layers
        hyperparams['training_params'] = {'batch_size': kwargs['batch_size'], 'epoch_overlap': kwargs['epoch_overlap'], 'num_steps': kwargs['num_steps'], 'learning_rate': kwargs['learning_rate'],
                                        'lambda_l2_reg': kwargs['lambda_l2_reg'], 'num_epochs': kwargs['num_epochs']}
        hyperparams['drop_out'] = {'input_prob': kwargs['input_prob'], 'output_prob': kwargs['output_prob'], 'state_prob': kwargs['state_prob']}
        hyperparams['early_stop'] = {'val_ratio': kwargs['val_ratio'], 'max_checks_without_progress': kwargs['max_checks_without_progress'], 'epoch_before_val': kwargs['epoch_before_val']}
        hyperparams['MSE_val'] = MSE_result[ind]
        return(hyperparams, kwargs['save_location'], prediction_train, prediction_val, prediction_test, train_loss_final, val_loss_final, test_loss_final)

