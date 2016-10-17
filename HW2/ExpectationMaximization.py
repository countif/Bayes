import pandas as pd
import numpy as np
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
        self.logLikelihoods = []
        K = set(self.y)


    def fit(self):
    
        def _var(x,w):
            return (-x.T*w)/self.sgm
        
        def _eStep(self):
            
                
            for i in range(N):
                
                _varValue = _var(X[i],w)
                
                if y[i] == 1:
                    e_phi[i] = _varValue + (sgm * (norm.pdf(_varValue) / (1 - norm.cdf(_varValue))))
                
                else:
                    e_phi[i] = _varValue + (sgm * (-norm.pdf(_varValue) / norm.cdf(_varValue)))
                    
        
        def _mStep(self):
            w = np.linalg.inv((lmbd + X.dot(X.T).sum())/sgm**2) * (X.dot(e_phi)/sgm**2)
            
        def _logLikelihood(self):
            _varValue = _var(self.X,np.tile(w, N)

            logLikelihood = (d/2 * np.log(lmbd/(2*math.pi))) - (lmbd/2*(w.T.dot(w))) + (self.y.dot(np.log(norm.cdf(_varValue)))).sum() + ((1 - self.y).dot(np.log(1 - norm.cdf(_varValue)))).sum()

            return logLikelihood

        for t in range(n_iter):
            _eStep()
            _mStep()
            self.logLikelihoods.append(_logLikelihood())



        

    def predict_proba(self,Xtest):
        
        self.Xtest = np.asarray(Xtest)
        _probit = np.zeros((len(Xtest),len(K)))
        predict_proba = np.zeros((len(Xtest),len(K)))
        
        def _probitDistribution(self,_varValue,y):
            np.power(norm.cdf(_varValue),y) * np.power((1 - norm.cdf(_varValue)),(1-y))
        
        for i in range(N):
            _varValue = _var(self.Xtest[i],w)
            
            for k in K:
                _probit[i][k] = _probitDistribution(_varValue,k)

            for k in K:
                predict_proba[i][k] = _probit[i][k]/np.sum(_probit[i])    

        return predict_proba


    def predict(self,Xtest):
        predict_proba = self.predict_proba(Xtest)
        predict = []
        for i in range(len(Xtest)):
            predict.append(np.argmax(predict_proba[i]))
             
        return predict


