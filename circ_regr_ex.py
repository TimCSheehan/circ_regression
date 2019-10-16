import numpy as np
from sklearn import linear_model

""" 
This code is meant to illustrate how circuluar regression is performed. Circ reg and circ mean are included for
convienience. NB: inputs are not taken in the same format. Uploaded on 10/16/19 to accompany SfN presentation.
Cleaner updates to follow.

Timothy Sheehan 10/19
"""

pi=np.pi

def circ_regr(Xtrain,Ytrain,Xtest,want_vec_len=False):
    # Xtrain - samples x nvox
    # Ytrain - samples [0,180]
    
    assert Xtrain.shape[0] == len(Ytrain), 'Not matching sizes'
    Ytrain_ang = Ytrain/180*pi*2
    Ytrain_mat = np.stack((np.sin(Ytrain_ang),np.cos(Ytrain_ang))).T
    regr = linear_model.LinearRegression().fit(Xtrain,Ytrain_mat)
    out = regr.predict(Xtest)
    ang_hat = np.arctan(out[:,0]/out[:,1])
    
    inds = np.where(out[:,1]<0)[0]
    ang_hat[inds] = ang_hat[inds]+pi
    ang_hat = np.mod(ang_hat,2*pi)*180/pi/2
    if want_vec_len:
        vec_len = np.sqrt(np.sum(np.power(out,2),1)) # should indicate confidence?
        return ang_hat,vec_len
    return ang_hat

def circ_corr_coef(x, y):
    """ 
    Calculate correlation coefficient between two circular variables
    Using Fisher & Lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0,2pi]
    """
    if np.any(x>90): # assume both variable range from [0,180]
        x = x/90*pi
        y = y/90*pi
    if np.any(x<0) or np.any(x>2*pi) or np.any(y<0) or np.any(y>2*pi):
        raise ValueError('x and y values must be between 0-2pi')
    n = np.size(x);
    assert(np.size(y)==n)
    A = np.sum(np.cos(x)*np.cos(y));
    B = np.sum(np.sin(x)*np.sin(y));
    C = np.sum(np.cos(x)*np.sin(y));
    D = np.sum(np.sin(x)*np.cos(y));
    E = np.sum(np.cos(2*x));
    Fl = np.sum(np.sin(2*x));
    G = np.sum(np.cos(2*y));
    H = np.sum(np.sin(2*y));
    corr_coef = 4*(A*B-C*D) / np.sqrt((np.power(n,2) - np.power(E,2) - np.power(Fl,2))*(np.power(n,2) - np.power(G,2) - np.power(H,2)));
    return corr_coef


def circ_mean(w,alpha=None):
    # w- should be samples x degrees
    # alpha - defaults to 180 samples +/- pi
    
    if not alpha:
        alpha = np.linspace(0,2*np.pi,181)
        alpha = alpha[:-1]
    t = w * np.exp(1j * alpha)
    r = np.sum(t, axis=1)
    mu = np.angle(r)
    conf = np.abs(r)
    est = mu/np.pi*90
    
    return est,conf