"""
Original work by Weike (Vicky) Sun vickysun@mit.edu/weike.sun93@gmail.com
Modified by Pedro Seber
"""
import numpy as np
import ace_R
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import math
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.api as sms
import statsmodels.api as sm
import scipy.stats as stats
import nonlinear_regression as nr
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge

import matplotlib.style
import matplotlib as mpl
mpl.style.use('default')

def nonlinearity_assess(X, y, plot, cat=None, alpha = 0.01, difference = 0.4, xticks = None, yticks = None, round_number = 0):
    """
    This function assesses the nonlinear correlation between X[:] and y
    
    Input: 
        X: independent variables of size N x m
        y: dependent variable of size N x 1
        plot: flag for plotting
        alpha: significance level for quaratic testing
        difference: significance level for maximal correlation - linear correlation
    
    Output:
        int, whether there is nonlinearity in dataset
    """
    Bi, _  = nr.poly_feature(X, degree = 2, interaction = True, power = False)
    Bi = Bi[:,X.shape[1]:]
  
    # Nonlinearity by linear correlation, quadratic test, and maximal correlation
    m = np.shape(X)[1]
    N = np.shape(X)[0]
    
    if plot:
        print('=== Scatter plot of the dataset ===')
        # Visualize the data
        dataset = np.concatenate((X,y.reshape((-1,1))), axis = 1)
        
        if xticks is None:
            xticks = [r'x$_'+str(i)+'$' for i in range(1,np.shape(X)[1]+1)]
        if yticks is None:
            yticks = ['y']
        name = xticks[:] + yticks[:]
        dataset=pd.DataFrame(data= dataset, columns =name)

        if m <= 10:
            plt.figure(figsize=(X.shape[1]*2,X.shape[1]*2)) 
            sns.set(font_scale=1.5)
            sns.pairplot(dataset)
            plt.savefig('pairplot_' + str(round_number)+'.png',dpi = 600,bbox_inches='tight')
        
        # Compute the correlation matrix
        corr = dataset.corr()
        corr[abs(corr)<1e-6] = 0    
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(X.shape[1]+1,X.shape[1]+1))
        # Draw the heatmap with the mask and correct aspect ratio
        s=17
        sns.set(font_scale=1.3) 
        plt.tick_params(labelsize=s)
        sns.heatmap(corr, cmap='RdBu',square=True,vmin=-1,vmax=1,linecolor="white", linewidths=0.8, ax=ax,annot=True,cbar_kws={"shrink": .82})
        plt.savefig('corrplot'+ str(round_number)+'.png', dpi = 600,bbox_inches='tight')
 
    # Pre-processing the data
    scaler_x = StandardScaler()
    scaler_x.fit(X)
    X = scaler_x.transform(X)   
    scaler_y = StandardScaler()
    scaler_y.fit(y)
    y=scaler_y.transform(y)
    
    if m > 1:
        scaler_B = StandardScaler()
        scaler_B.fit(Bi)
        Bi=scaler_B.transform(Bi)

    LC = np.zeros((m,1))
    QT = np.zeros((m,1))
    MC = np.zeros((m,1))
    
    for i in range(m):
        # Linear correlation
        LC[i] = np.corrcoef(X[:,i],y.squeeze())[0,1]

        # Quaratic test
        reg = LinearRegression(fit_intercept=False).fit(X[:,i].reshape(-1, 1), y.reshape(-1, 1))
        y_pred = reg.predict(X[:,i].reshape(-1, 1))
        mse1 = np.sum((y.reshape(-1, 1)-y_pred)**2)
        regq = LinearRegression(fit_intercept=False).fit(np.array([X[:,i]**2, X[:,i]]).transpose(), y.reshape(-1, 1))
        yq_pred = regq.predict(np.array([X[:,i]**2, X[:,i]]).transpose())
        mse2 = np.sum((y.reshape(-1, 1)-yq_pred)**2)
        F = (mse1- mse2)/(mse2/(N-2))
        p_value = 1 - f.cdf(F, 1, N-2)
        QT[i] = 0 if p_value < 10*np.finfo(float).eps else p_value
                
        # Maximal correlation by ACE algorithm
        if cat is None or cat[i] == 0:
            MC[i] = ace_R.ace_R(X[:,i].reshape(-1, 1), y)
        else:
            MC[i] = ace_R.ace_R(X[:,i].reshape(-1, 1), y, cat=1)
    
    # Bilinear 
    if m > 1:
        p_values = np.zeros((Bi.shape[1],1))
        bi_test_threshold = alpha/np.shape(p_values)[0]
        counter = 0
        for i in range(m-1):
            for j in range(i+1,m):
                regl = LinearRegression(fit_intercept=False).fit(np.array([X[:,i], X[:,j]]).transpose(), y.reshape(-1,1))
                yl_pred = regl.predict(np.array([X[:,i], X[:,j]]).transpose())
                mse1 = np.sum((y.reshape(-1,1) - yl_pred)**2)
                regi = LinearRegression(fit_intercept=False).fit(np.array([X[:,i], X[:,j],Bi[:, counter]]).transpose(), y.reshape(-1,1))
                yi_pred = regi.predict(np.array([X[:,i], X[:,j], Bi[:,counter]]).transpose())
                mse2 = np.sum((y.reshape(-1,1)-yi_pred)**2)
                counter += 1
    
                F = (mse1-mse2)/(mse2/(N-2))
                p_values[counter-1] = 1-f.cdf(F, 1, N-2)
            
     
        tri = np.zeros((m-1, m-1))
        count = 0
        for i in range(1,m):
            if i == 1:
                tri[-i, -1] = p_values[-i-count:]
            else:
                tri[-i, -i:] =  p_values[-i-count:-count].flatten()
            count += i  
        
        tri[tri<1e-15] = 0
    
    # Calculate test threshold
    q_test_threshold = alpha/np.shape(QT)[0]
    
    if plot:
        print('=== Nonlinearity test results ===')
        # Plot for linear correlation
        cmap = sns.diverging_palette(10,250, as_cmap=True)
        if xticks is None:
            xticks = [r'x$_'+str(i)+'$' for i in range(1,np.shape(X)[1]+1)]
        plt.figure(figsize=(X.shape[1],3))
        sns.set(font_scale=1.6)
        sns.set_style("whitegrid")
        ax=sns.heatmap(LC.transpose(),linewidths=0.8,vmin=-1,vmax=1,cmap=cmap,annot=True,\
                       linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,\
                       yticklabels=yticks, cbar_kws={'label': 'linear correlation',"orientation": "horizontal",'ticks' : [-1,0,1]}) 
        loc, labels = plt.yticks()
        ax.set_yticklabels(labels, rotation=0)
        plt.savefig('linear_correlation_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
        
        # Plot the quadratic test results
        plt.figure(figsize=(X.shape[1],3))
        # Calcaultate the rejection threhsold (default alpha=0.01 for one test)
        plot_threshold = math.floor(np.log10(q_test_threshold))
        plot_threshold = 10**plot_threshold
        # Set lower bar
        low_value_flags = QT < plot_threshold**2
        QT[low_value_flags] = plot_threshold**2
        ax=sns.heatmap(QT.transpose(),linewidths=0.8,vmin=plot_threshold**2,vmax=1,cmap="Blues",annot=True, norm=LogNorm(),\
                       linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,yticklabels=yticks,\
                       cbar_kws={'label': 'p-value of quadratic test',"orientation": "horizontal",'ticks' : [plot_threshold**2,plot_threshold,1]}) 
        loc, labels = plt.yticks()
        ax.set_yticklabels(labels, rotation=0)
        plt.savefig('quaradtic_test_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')

        # Plot maximal correlation
        plt.figure(figsize=(X.shape[1],3))
        ax=sns.heatmap(MC.transpose(),linewidths=0.8,vmin=0,vmax=1,cmap="Blues",annot=True,\
                       linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,yticklabels=yticks,\
                       cbar_kws={'label': 'maximal correlation',"orientation": "horizontal",'ticks' : [0,0.5,1]}) 
        loc, labels = plt.yticks()
        ax.set_yticklabels(labels, rotation=0)
        plt.savefig('maximal_correlation_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
        
        # Bilinear term
        if m>1:
        # Generate a mask for the upper triangle
            mask = np.zeros_like(tri, dtype=np.bool)
            mask[np.tril_indices_from(mask, k=-1)] = True
            s=17
    
            # Set up the matplotlib figure
            sns.set_style("white")
            fig, ax = plt.subplots(figsize=(2*(m-1),2*(m-1)))
            sns.set(font_scale=1.3)
            plt.tick_params(labelsize=s)
            plot_threshold = 0.15
            sns.heatmap(tri, cmap="Blues", mask=mask,square=True,vmin=0,vmax=1,linecolor="white", linewidths=0.8, ax=ax,annot=True,cbar_kws={"shrink": .82, 'ticks' : [0, 0.15, 0.5, 1]})
            ax.set_xticklabels(xticks[1:])
            ax.set_yticklabels(xticks[:-1])
            plt.title('p_values for bilinear terms')
            plt.savefig('f_bilinear_'+ str(round_number)+'.png', dpi = 600,bbox_inches='tight')

    # Detemine whether nonlinearity is significant
    corr_difference = MC - abs(LC) > difference # Default 0.4, for maximal correlation
    corr_absolute   = [a and b for a,b in zip(MC > 0.92, MC - abs(LC)>0.1)]
    corr_difference = corr_absolute or corr_difference
    q_test = QT<q_test_threshold # For quadratic test
    if m > 1:
        bi_test = p_values < bi_test_threshold
        overall_result = np.concatenate((corr_difference, q_test, bi_test), axis=0) # overall result
        return int(True in overall_result) # True = nonlinear correlation
    else:
        overall_result = np.concatenate((corr_difference, q_test), axis=0) # overall result
        return int(True in overall_result)

def collinearity_assess(X, y, plot, xticks = None , yticks = None, round_number = 0):
    """
    This funcion assesses collinearity in the independent variables, using the variation inflation factor
    Rule of thumb: if VIF > 5, then the explanatory variable is highly collinear with other
    explanatory variables, leading to large standard errors in the parameter estimates.
    
    Input: 
        X: independent variables of size N x m
        y: dependent variable of size N x 1
        plot: flag for plotting
    
    Output:
        int, whether there is collinearity in the independent variable
    """
    scaler_x = StandardScaler()
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    scaler_y = StandardScaler()
    scaler_y.fit(y)
    y=scaler_y.transform(y)
    
    if np.shape(X)[1] == 1:
        # Univariate regression problem does not suffer from colinearity
        return 0
    elif np.shape(X)[1] > np.shape(X)[0]:
        return 1
    else:
        VIF = [variance_inflation_factor(X, i) for i in range(0,np.shape(X)[1])]
        if plot:
            print('=== Multicollinearity Results ===')
            if xticks is None:
                xticks = [r'x$_'+str(i)+'$' for i in range(1,np.shape(X)[1]+1)]
            if yticks is None:
                yticks = ['y']
            plt.figure(figsize=(X.shape[1],3))
            sns.set(font_scale=1.6)
            sns.set_style("whitegrid")
            ax=sns.heatmap(np.array(VIF).reshape(1,-1),linewidths=0.8,vmin=1,vmax=10,cmap='Blues',annot=True,\
                               linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,\
                               yticklabels=yticks, cbar_kws={'label': 'variance inflation factor',"orientation": "horizontal",'ticks' : [1, 5, 10]}) 
            plt.savefig('VIF_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
            
        # Check whether VIF > 5
        for i in range(0,np.shape(X)[1]):
            if VIF[i] > 5:
                return int(True)
        return int(False)
        
def dynamic_assess(x, plot=1, y = None, round_number = 0, alpha = 0.01, freq = 1):
    """
    x: time series data of interests of size Nx1
    y: another time series for calculating corss-correlation of size Nx1
    plot: flag for plotting
    alpha: significance level
    freq: sampling frequency of the time series Hz
    
    Output:
        significant lags for ACF, PACF and CCF
    """
    # Dickey-Fuller Tests for statinonary, below alpha is stationary
    x = x.flatten()
    xdf = sm.tsa.stattools.adfuller(x,1)
    if xdf[1] > alpha:
        print('x is not stationary')
        
    if y is not None:
        ydf = sm.tsa.stattools.adfuller(y, 1)
        if ydf[1] < alpha:
            print('y is not stationary')        
    # ACF
    [acf, confint, qstat, acf_pvalues] = sm.tsa.stattools.acf(x, qstat = True, alpha = alpha)
    acf_detection = acf_pvalues < alpha # Ljung-Box Q-Statistic
    acf_lag = [i for i,u in enumerate(acf_detection) if u == True] 

    # PACF
    [pacf, confint_pacf] = sm.tsa.stattools.pacf(x, alpha = alpha)
    pacf_lag = [i for i,u in enumerate(pacf) if abs(u)>1.96/np.sqrt(x.shape[0])]
   
    # CCF
    if y is not None:
        plt.figure(figsize=(5,3))
        plt.xcorr(x,y, normed = True, usevlines=True, maxlags=20)
        plt.axhline(y=2.575*1/np.sqrt(x.shape[0]), color='blue', linestyle='--',alpha=0.9) # 99% confidence interval
        plt.axhline(y=-2.575*1/np.sqrt(x.shape[0]), color='blue', linestyle='--',alpha=0.9) # 99% confidence interval
        font = 15
        plt.title('Cross-correlation plot',fontsize=font)
        plt.xlabel('Lag',fontsize=font)
        plt.tick_params(labelsize=font-1)
        plt.tight_layout()
        plt.savefig('CCF_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
    
    # FFT
    x=x.squeeze()
    if x.shape[0] % 2 != 0:
        x = x[:-1]
    L = x.shape[0]
    x_fft = np.fft.fft(x)
    P2 = abs(x_fft/L)
    P1 = P2[0:int(L/2+1)]
    P11 = P1[:]
    P1[1:-1] = 2*P11[1:-1]
    f = freq*np.linspace(0, int(L/2), num=int(L/2)+1,endpoint = True)/ L
    plt.figure(figsize=(5,3))
    plt.plot(f,P1)
    font = 15
    plt.title('Single-sided amplitude spectrum',fontsize = font)
    plt.xlabel('frequncy (Hz)', fontsize = font)
    plt.ylabel('|P1(f)|', fontsize = font)
    plt.tight_layout()
    plt.savefig('FFT_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')  
    return (acf_lag, pacf_lag)
  
def residual_analysis(X, y, y_hat, nlag =None, alpha = 0.01, round_number = 0):
    """
    This funcion assesses the residuals (heteroscedasticity and dyanmics)
    Heteroscedasticity is tested on Breusch-Pagan Test and White Test
    Dyanmics is assessed based on ACF and PACF
    
    Input: 
        X: independent variables of size N x m
        y_hat: fitted dependent variable of size N x 1
        residual: residuals of size N x 1
        alpha: significance level for statistical tests

    Output:
        figures, residual analysis
        (int_heteroscedasticity, int_dynamics), whether there is heteroscedasticity and dynamics
    """
    print('=== Residual Analysis ===')
    residual = y-y_hat
    if nlag is None:
        if y.shape[0] < 40:
            #nlag = 10
            nlag = y.shape[0]//2 - 1 # Apparently nlag must be < y.shape[0], else sm.graphics.tsa.plot_acf() returns an error. It must also be < 0.5*y.shape[0], else sm.graphics.tsa.plot_pacf() returns an error.
        elif y.shape[0] > 200:
            nlag = 50
        else:
            nlag = y.shape[0]//4
            
    # Basic Residual Plot
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    plt.plot(y,y_hat,'*')
    sm.qqline(ax=ax, line='45', fmt='k--')
    plt.ylabel('fitted y', fontsize=14)
    plt.xlabel('y', fontsize=14)
    plt.axis('scaled')
    plt.tight_layout()
    plt.title('Real vs Fitted')
    plt.savefig('Fit_plot_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')

    fontsize = 20
    markersize = 8
    sample_number = np.linspace(1, residual.shape[0], residual.shape[0], endpoint=True)
    fig, axs = plt.subplots(2, 2, figsize=(12,9))
    axs[0,0].hist(residual, density = True, facecolor='skyblue', alpha=1, edgecolor='black')
    axs[0,0].axvline(x=0, color='k', linestyle='--',alpha=0.6)
    axs[0,0].set_ylabel('Frequency',fontsize = fontsize)
    axs[0,0].set_xlabel('Residual',fontsize = fontsize)
    axs[0,0].set_title('Residual histogram',fontsize = fontsize)
    axs[0,0].tick_params(labelsize = fontsize-3)
       
    axs[0,1].plot(sample_number, residual, 'o', color = 'cornflowerblue', markersize = markersize)
    axs[0,1].axhline(y=0, color='k', linestyle='--',alpha=0.6)
    axs[0,1].set_xlabel('Sample number',fontsize = fontsize)
    axs[0,1].set_ylabel('Residual',fontsize = fontsize)
    axs[0,1].set_title('Residual',fontsize = fontsize)
    axs[0,1].tick_params(labelsize = fontsize-3)

    sm.qqplot(residual.squeeze(), stats.t, fit=True,ax=axs[1,0])
    sm.qqline(ax=axs[1,0], line='45', fmt='k--')
    axs[1,0].set_xlabel('Theoretical quantiles',fontsize = fontsize)
    axs[1,0].set_ylabel('Sample quantiles',fontsize = fontsize)
    axs[1,0].set_title('Normal Q-Q plot',fontsize = fontsize)
    axs[1,0].tick_params(labelsize = fontsize-3)
    axs[1,0].get_lines()[0].set_markersize(markersize)
    axs[1,0].get_lines()[0].set_markerfacecolor('cornflowerblue')
    
    axs[1,1].plot(y_hat, residual, 'o', color = 'cornflowerblue', markersize = markersize)
    axs[1,1].axhline(y=0, color='k', linestyle='--',alpha=0.6)
    axs[1,1].set_xlabel('Fitted response',fontsize = fontsize)
    axs[1,1].set_ylabel('Residual',fontsize = fontsize)
    axs[1,1].set_title('Residual versus fitted response',fontsize = fontsize)
    axs[1,1].tick_params(labelsize = fontsize-3)
    plt.tight_layout()
    plt.savefig('Residual_plot_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')

    # Heteroscedaticity
    # Test whether variance is the same in 2 subsamples
    test_GF = sms.het_goldfeldquandt(residual,X)
    name = ['F statistic', 'p-value']
    GF_test = dict(zip(name,test_GF[0:2]))
    # Breusch-Pagan test for heteroscedasticity
    test_BP = sms.het_breuschpagan(residual,np.column_stack((np.ones((y_hat.shape[0],1)),y_hat)))
    BP_test = dict(zip(name,test_BP[2:]))
    # White test for heteroscedasticity
    test_white = sms.het_white(residual, np.column_stack((np.ones((y_hat.shape[0],1)),y_hat)))
    White_test = dict(zip(name,test_white[2:]))
    
    int_heteroscedasticity = 1
    if test_GF[1] > alpha and test_BP[-1] > alpha and test_white[-1] > alpha:
        int_heteroscedasticity = 0
    
    # Dynamics
    # Autocorrelation
    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(111)    
    fig = sm.graphics.tsa.plot_acf(residual, lags=nlag, ax=ax1, alpha= alpha)
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(14)
    ax1.set_xlabel('Lag')
    plt.tight_layout()
    plt.savefig('ACF_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
    
    # Partial autocorrelation
    fig = plt.figure(figsize=(5,3))
    ax2 = fig.add_subplot(111)    
    fig = sm.graphics.tsa.plot_pacf(residual, lags=nlag, ax=ax2, alpha= alpha)
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(14)
    ax2.set_xlabel('Lag')
    plt.tight_layout()
    plt.savefig('PACF_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
    
    # ACF
    [acf, confint, qstat, acf_pvalues] = sm.tsa.stattools.acf(residual, nlags=nlag,qstat = True, alpha = alpha)
    acf_detection = acf_pvalues < (alpha/nlag) # Ljung-Box Q-Statistic
    acf_lag = [i for i,x in enumerate(acf_detection) if x == True] 
      
    # PACF
    [pacf, confint_pacf] = sm.tsa.stattools.pacf(residual, nlags=nlag, alpha = alpha)
    pacf_lag = [i for i,x in enumerate(pacf) if x<confint_pacf[i][0] or x>confint_pacf[i][1]]
    
    if acf_lag != [] or pacf_lag != []:
        int_dynamics = 1
    else:
        int_dynamics = 0
        
    return (int_heteroscedasticity, int_dynamics)

def nonlinearity_assess_dynamic(X, y, plot, cat=None, alpha = 0.01, difference = 0.4, xticks = None, yticks = None, round_number = 0, lag = 3):
    """
    This function assesses the nonlinear correlation between X[:] and y
    
    Input: 
        X: independent variables of size N x m
        y: dependent variable of size N x 1
        plot: flag for plotting
        alpha: significance level for quaratic testing
        difference: significance level for maximal correlation - linear correlation
    
    Output:
        int, whether there is nonlinearity in dataset
    """
    # Nonlinearity by linear correlation, quadratic test, and maximal correlation
    m = np.shape(X)[1]
    N = np.shape(X)[0]
    if yticks is None:
        yticks = ['y']
    if xticks is None:
        xticks = [r'x$_'+str(i)+'$' for i in range(1,np.shape(X)[1]+1)]
    xticks = xticks + yticks
    ylabel = ['lag'+str(i+1) for i in range(lag)]
    
    # Pre-processing the data
    scaler_x = StandardScaler()
    scaler_x.fit(X)
    X = scaler_x.transform(X)
    scaler_y = StandardScaler()
    scaler_y.fit(y)
    y=scaler_y.transform(y)
    
    LC = np.zeros((m+1,lag))
    QT = np.zeros((m+1,lag))
    MC = np.zeros((m+1,lag))
    
    for l in range(lag):
        for i in range(m):
            # Linear correlation
            LC[i,l] = np.corrcoef(X[:-l-1,i],y[l+1:].squeeze())[0,1]
            
            # Quadratic test
            reg = LinearRegression(fit_intercept=False).fit(X[:-l-1,i].reshape(-1, 1), y[l+1:].reshape(-1, 1))
            y_pred = reg.predict(X[:-l-1,i].reshape(-1, 1))
            mse1 = np.sum((y[l+1:].reshape(-1, 1)-y_pred)**2)
            regq = LinearRegression(fit_intercept=False).fit(np.array([X[:-l-1,i]**2,X[:-l-1,i]]).transpose(), y[l+1:].reshape(-1, 1))
            yq_pred = regq.predict(np.array([X[:-l-1,i]**2, X[:-l-1,i]]).transpose())
            mse2 = np.sum((y[l+1:].reshape(-1, 1)-yq_pred)**2)
            F = (mse1- mse2)/(mse2/(N-2))
            p_value = 1 - f.cdf(F, 1, N-2)
            QT[i,l] = 0 if p_value < 10*np.finfo(float).eps else p_value
                    
            # Maximal correlation by ACE algorithm
            if cat is None or cat[i] == 0:
                MC[i,l] = ace_R.ace_R(X[:-l-1,i].reshape(-1, 1), y[l+1:])
            else:
                MC[i,l] = ace_R.ace_R(X[:-l-1,i].reshape(-1, 1), y[l+1:], cat=1)

    for l in range(lag):
        # Linear correlation
        LC[m,l] = np.corrcoef(y[:-l-1].squeeze(),y[l+1:].squeeze())[0,1]
            
        # Quadratic test
        reg = LinearRegression(fit_intercept=False).fit(y[:-l-1].reshape(-1, 1), y[l+1:].reshape(-1, 1))
        y_pred = reg.predict(y[:-l-1].reshape(-1, 1))
        mse1 = np.sum((y[l+1:].reshape(-1, 1)-y_pred)**2)
        regq = LinearRegression(fit_intercept=False).fit(np.array([y[:-l-1].squeeze()**2,y[:-l-1].squeeze()]).transpose(), y[l+1:].reshape(-1, 1))
        yq_pred = regq.predict(np.array([y[:-l-1].squeeze()**2, y[:-l-1].squeeze()]).transpose())
        mse2 = np.sum((y[l+1:].reshape(-1, 1)-yq_pred)**2)
        F = (mse1- mse2)/(mse2/(N-2))
        p_value = 1 - f.cdf(F, 1, N-2)
        QT[m,l] = 0 if p_value < 10*np.finfo(float).eps else p_value
                    
        # Maximal correlation by ACE algorithm
        if cat is None or cat[i] == 0:
            MC[m,l] = ace_R.ace_R(y[:-l-1].reshape(-1, 1), y[l+1:])
        else:
            MC[m,l] = ace_R.ace_R(y[:-l-1].reshape(-1, 1), y[l+1:], cat=1)
        
    if plot:
        print('=== Nonlinearity test results for lagged data ===')
        # Plot for linear correlation
        cmap = sns.diverging_palette(10,250, as_cmap=True)
        plt.figure(figsize=(X.shape[1]+1,lag))
        sns.set(font_scale=1.6)
        sns.set_style("whitegrid")
        ax=sns.heatmap(LC.transpose(),linewidths=0.8,vmin=-1,vmax=1,cmap=cmap,annot=True,\
                       linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,\
                       yticklabels=ylabel, cbar_kws={'label': 'linear correlation',"orientation": "horizontal",'ticks' : [-1,0,1]}) 
        loc, labels = plt.yticks()
        ax.set_yticklabels(labels, rotation=0)
        plt.savefig('linear_correlation_' + str(round_number)+ 'lag'+str(lag)+'.png', dpi = 600,bbox_inches='tight')
        
        # Plot quadratic test
        plt.figure(figsize=(X.shape[1]+1,lag))
        # Calculate the rejection threhsold (default alpha=0.01 for one test)
        q_test_threshold = alpha/np.shape(QT)[0]/np.shape(QT)[1]
        plot_threshold = math.floor(np.log10(q_test_threshold))
        plot_threshold = 10**plot_threshold
        # Set lower bar
        low_value_flags = QT < plot_threshold**2
        QT[low_value_flags] = plot_threshold**2
        ax=sns.heatmap(QT.transpose(),linewidths=0.8,vmin=plot_threshold**2,vmax=1,cmap="Blues",annot=True, norm=LogNorm(),\
                       linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,yticklabels=ylabel,\
                       cbar_kws={'label': 'p-value of quadratic test',"orientation": "horizontal",'ticks' : [plot_threshold**2,plot_threshold,1]}) 
        loc, labels = plt.yticks()
        ax.set_yticklabels(labels, rotation=0)
        plt.savefig('quaradtic_test_' + str(round_number)+ 'lag'+str(lag)+'.png', dpi = 600,bbox_inches='tight')

        # Plot maximal correlation
        plt.figure(figsize=(X.shape[1]+1,lag))
        ax=sns.heatmap(MC.transpose(),linewidths=0.8,vmin=0,vmax=1,cmap="Blues",annot=True,\
                       linecolor="white",annot_kws={"size": 14},xticklabels=xticks,square=True,yticklabels=ylabel,\
                       cbar_kws={'label': 'maximal correlation',"orientation": "horizontal",'ticks' : [0,0.5,1]}) 
        loc, labels = plt.yticks()
        ax.set_yticklabels(labels, rotation=0)
        plt.savefig('maximal_correlation_' + str(round_number)+ 'lag'+str(lag)+'.png', dpi = 600,bbox_inches='tight')
    
    if m > 1:
        # For quadratic test
        Bi, _  = nr.poly_feature(X, degree = 2, interaction = True, power = False)
        Bi = Bi[:,X.shape[1]:]
        bi_test_result = np.zeros(l+1)
        for l in range(lag):
            p_values = np.zeros((Bi.shape[1],1))
            counter = 0
            for i in range(m-1):
                for j in range(i+1,m):
                    regl = LinearRegression(fit_intercept=False).fit(np.array([X[:-l-1,i], X[:-l-1,j]]).transpose(), y[l+1:].reshape(-1,1))
                    yl_pred = regl.predict(np.array([X[:-l-1,i], X[:-l-1,j]]).transpose())
                    mse1 = np.sum((y[l+1:].reshape(-1,1) - yl_pred)**2)
                    regi = LinearRegression(fit_intercept=False).fit(np.array([X[:-l-1,i], X[:-l-1,j],Bi[:-l-1, counter]]).transpose(), y[l+1:].reshape(-1,1))
                    yi_pred = regi.predict(np.array([X[:-l-1,i], X[:-l-1,j], Bi[:-l-1,counter]]).transpose())
                    mse2 = np.sum((y[l+1:].reshape(-1,1)-yi_pred)**2)
                    counter += 1
                    F = (mse1-mse2)/(mse2/(N-2))
                    p_values[counter-1] = 1-f.cdf(F, 1, N-2)
         
            tri = np.zeros((m-1, m-1))
            count = 0
            for i in range(1,m):
                if i == 1:
                    tri[-i, -1] = p_values[-i-count:]
                else:
                    tri[-i, -i:] =  p_values[-i-count:-count].flatten()
                count += i  
            tri[tri<1e-15] = 0
            bi_test_result[l] = sum(p_values < alpha/np.shape(p_values)[0]/(lag+1))
            
            if plot:
                mask = np.zeros_like(tri, dtype=np.bool)
                mask[np.tril_indices_from(mask, k=-1)] = True
                s=17
                # Set up the matplotlib figure
                sns.set_style("white")
                fig, ax = plt.subplots(figsize=(1.5*(m-1),1.5*(m-1)))
                sns.set(font_scale=1.3)
                plt.tick_params(labelsize=s)
                plot_threshold = 0.15
                sns.heatmap(tri, cmap="Blues", mask=mask,square=True,vmin=0,vmax=1,linecolor="white", linewidths=0.8, ax=ax,annot=True,cbar_kws={"shrink": .82, 'ticks' : [0, 0.15, 0.5, 1]})
                ax.set_xticklabels(xticks[1:])
                ax.set_yticklabels(xticks[:-1])
                plt.title('p_values for bilinear terms lag ' + str(l))
                plt.savefig('f_bilinear_'+ str(round_number)+ 'lag'+str(l)+'.png', dpi = 600,bbox_inches='tight')

        bi_test = sum(bi_test_result) > 1

    # Detemine whether nonlinearity is significant
    corr_difference = MC - abs(LC) > difference # Default 0.4, for maximal correlation
    a= MC > 0.92
    b = MC-abs(LC) > 0.1
    corr_absolute = a*b
    corr_difference = corr_absolute + corr_difference
    q_test_threshold = alpha/np.shape(QT)[0]/np.shape(QT)[1]
    q_test = QT<q_test_threshold # For quadratic test
    overall_result = np.concatenate((corr_difference, q_test), axis=0) # Overall result
    
    if m > 1:
        return int(True in overall_result.flatten() or bi_test) # True = nonlinear correlation
    else:
        return int(True in overall_result.flatten())

