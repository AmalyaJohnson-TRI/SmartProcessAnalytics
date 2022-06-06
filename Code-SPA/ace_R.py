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
ace = importr('acepack')

def ace_R(x, y, cat = None):
    '''
    x: one predictor, Nx1
    y: one response, Nx1
    categorical:wheter variables are categorical, [y, x], integer vector
    '''
    
    # Data preparation
    rpy2.robjects.numpy2ri.activate()
    
    # Convert training data numpy to R vector/matrix
    nrx,ncx = x.shape
    xr = ro.r.matrix(x,nrow=nrx,ncol=ncx)
    ro.r.assign("x", xr)
    nry,ncy = y.shape
    yr = ro.r.matrix(y,nrow=nry,ncol=ncy)
    ro.r.assign("y", yr)
    
    # Calculate transformation
    if cat is not None:
        a = ace.ace(xr,yr, cat = 1)
    else:
        a = ace.ace(xr,yr)
        
    tx = a.rx2('tx')
    ty = a.rx2('ty')
    
    # Final correlation coefficient
    corr = np.array(ro.r.cor(tx,ty))[0][0]
    return corr

