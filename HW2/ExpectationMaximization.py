import numpy as np
from scipy.stats import norm
import math

class Probit():
    def __init__(self,X,y,sgm=1.5,lmbd=1,n_iter=100):

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.sgm = float(sgm)
        self.lmbd = lmbd
        self.n_iter = n_iter
        self.X_shape = np.shape(self.X)
        self.N = self.X_shape[0]
        self.d = self.X_shape[1]
        self.w = np.zeros((self.d))
        self.e_phi = np.zeros((self.N))
        self.logLikelihoods = []
        self.K = np.unique(self.y)

    
    def fit(self):
    
        
        
        def _eStep():
            
                
            for i in range(self.N):
                
                _xw = (self.X[i].T.dot(self.w))
                _xwSgm = (-1*_xw)/self.sgm
                
                if self.y[i] == 1:
                    self.e_phi[i] = _xw + (self.sgm * norm.pdf(_xwSgm)) / (1 - norm.cdf(_xwSgm))
                
                if self.y[i] == 0:
                    self.e_phi[i] = _xw + (self.sgm * -1 * norm.pdf(_xwSgm)) / norm.cdf(_xwSgm)
                    
        
        def _mStep():
            part1 = np.linalg.inv((self.lmbd + self.X.T.dot(self.X)/(self.sgm**2)))
            part2 = np.sum(np.multiply(self.X.T,self.e_phi).T,axis=0)/(self.sgm**2)
            
            self.w = part2.dot(part1)
            
        def _logLikelihood():
            
            logLikelihood_1 = (self.d/2 * np.log(self.lmbd/(2*math.pi)))
            logLikelihood_2 = (self.lmbd/2*(self.w.T.dot(self.w)))
            logLikelihood_3 = 0
            logLikelihood_4 = 0
            
            for i in range(self.N):
                _pxwSgm = (self.X[i].T.dot(self.w))/self.sgm
                logLikelihood_3 += self.y[i]*np.log(norm.cdf(_pxwSgm))
                logLikelihood_4 += (1 - self.y[i])*np.log(1 - norm.cdf(_pxwSgm))
            
            logLikelihood = logLikelihood_1 - logLikelihood_2 + logLikelihood_3 + logLikelihood_4
            
            return float(logLikelihood)

            
        for t in range(self.n_iter):
            print t
            _eStep()
            _mStep()
            
            self.logLikelihoods.append(_logLikelihood())
        
        
    def predict_proba(self,Xtest):
        
        self.Xtest = np.asarray(Xtest)
        _probit = np.zeros((len(Xtest),len(self.K)))
        predict_proba = np.zeros((len(Xtest),len(self.K)))
        
        
        for i in range(len(self.Xtest)):
            _xwSgm = (self.Xtest[i].T.dot(self.w))/self.sgm
            for k in self.K:
                _probit[i][k] = np.power(norm.cdf(_xwSgm),k) * np.power((1 - norm.cdf(_xwSgm)),1 - k)

            for k in self.K:
                predict_proba[i][k] = _probit[i][k]/np.sum(_probit[i])    

        return predict_proba


    def predict(self,Xtest):
        predict_proba = self.predict_proba(Xtest)
        predict = []
        for i in range(len(Xtest)):
            predict.append(np.argmax(predict_proba[i]))
             
        return predict


