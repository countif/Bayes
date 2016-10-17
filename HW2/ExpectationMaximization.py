import pandas as pd
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm
import math

class Probit():
    def __init__(self,X,y,sgm=1.5,lmbd=1,n_iter=100):

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.sgm = sgm
        self.lmbd = lmbd
        w = np.zeros()
        X_shape = np.shape(X)
        N = X_shape[0]
        d = X_shape[1]
        w = np.zeros((d))
        e_phi = np.zeros((N,d))

    def fit(self):
        
    def _eStep(self):
        
        def _var(x,w):
            return (-x.T*w)/self.sgm
            
        for i in range(N):
            
            _varValue = _variable(X[i],w)
            
            if y[i] == 1:
                e_phi[i] = _varValue + (sgm * (norm.pdf(_varValue) / (1 - norm.cdf(_varValue))))
            
            else:
                e_phi[i] = _varValue + (sgm * (-norm.pdf(_varValue) / norm.cdf(_varValue)))
                
    
    def _mStep(self):
        w = np.linalg.inv((lmbd + X.dot(X.T).sum())/sgm**2) * (X.dot(e_phi)/sgm**2)
        
    def _logLikelihood(self):

        

    def predict_proba(self,Xtest):
        

    def predict(self,Xtest):
        predict_proba = self.predict_proba(Xtest)
        predict = []
        for i in range(len(Xtest)):
            predict.append(np.argmax(predict_proba[i]))
             
        return predict


