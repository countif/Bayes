import numpy as np
from scipy.stats import norm
import math

class Probit():
    def __init__(self,X,y,sgm=1.5,lmbd=1,n_iter=100):

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.sgm = sgm
        self.lmbd = lmbd
        self.n_iter = n_iter
        self.X_shape = np.shape(self.X)
        self.N = self.X_shape[0]
        self.d = self.X_shape[1]
        self.w = np.zeros((self.d))
        self.e_phi = np.zeros((self.N,self.d))
        self.logLikelihoods = []
        self.K = np.unique(self.y)

    def _var(self,x,w):
        x = x*-1
        return (x.T.dot(w))/self.sgm

    def fit(self):
    
        
        
        def _eStep():
            
                
            for i in range(self.N):
                
                _varValue = self._var(self.X[i],self.w)
                
                if self.y[i] == 1:
                    self.e_phi[i] = self.X[i].T.dot(self.w) + (self.sgm * (norm.pdf(_varValue) / (1 - norm.cdf(_varValue))))
                
                else:
                    self.e_phi[i] = self.X[i].T.dot(self.w) + (self.sgm * (-norm.pdf(_varValue) / norm.cdf(_varValue)))
                    
        
        def _mStep():
            self.w = np.linalg.inv((self.lmbd + self.X.T.dot(self.X)/(self.sgm**2))).dot(np.sum((self.X * self.e_phi),axis=0)/(self.sgm**2))
        
            
        def _logLikelihood():
            
            logLikelihood_1 = (self.d/2 * np.log(self.lmbd/(2*math.pi)))
            logLikelihood_2 = (self.lmbd/2*(self.w.T.dot(self.w)))
            logLikelihood_3 = 0
            logLikelihood_4 = 0
            
            for i in range(self.N):
                _varValue = self._var(self.X[i],self.w)
                logLikelihood_3 += self.y[i]*np.log(norm.cdf(_varValue))
                logLikelihood_4 += (1 - self.y[i])*np.log(1 - norm.cdf(_varValue))
            
            logLikelihood = logLikelihood_1 - logLikelihood_2 + logLikelihood_3 + logLikelihood_4
            
            return logLikelihood

            
        for t in range(self.n_iter):
            print t
            _eStep()
            _mStep()
            #llh = _logLikelihood()
            #print llh
            #self.logLikelihoods.append(llh)
            

    def predict_proba(self,Xtest):
        
        self.Xtest = np.asarray(Xtest)
        _probit = np.zeros((len(Xtest),len(self.K)))
        predict_proba = np.zeros((len(Xtest),len(self.K)))
        
        def _probitDistribution(_varValue,y):
            np.power(norm.cdf(_varValue),y) * np.power((1 - norm.cdf(_varValue)),(1-y))
        
        for i in range(len(self.Xtest)):
            p_varValue = (self.Xtest[i].dot(self.w))/self.sgm
            
            for k in self.K:
                _probit[i][k] = _probitDistribution(p_varValue,k)

            for k in self.K:
                predict_proba[i][k] = _probit[i][k]/np.sum(_probit[i])    

        return predict_proba


    def predict(self,Xtest):
        predict_proba = self.predict_proba(Xtest)
        predict = []
        for i in range(len(Xtest)):
            predict.append(np.argmax(predict_proba[i]))
             
        return predict


