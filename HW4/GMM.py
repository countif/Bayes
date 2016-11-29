#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:39:46 2016

@author: lfawaz
"""
import numpy as np
from scipy.stats import multivariate_normal

class EM_GMM ():
    def __init__ (self,X,K=2,n_iter=100):
        self.X = np.asarray(X)
        self.n_iter = n_iter
        self.X_shape = np.shape(self.X)
        self.N = self.X_shape[0]
        self.d = self.X_shape[1]        
        self.logLikelihoods = []
        self.K = K
        self.phi = np.zeros((self.N,self.K))
        self.pi = np.random.normal((1,self.K))
        self.pi = self.pi/np.sum(self.pi)
        self.mu = np.random.normal(size=(self.K,self.d))
        self.sgm = np.random.uniform(size=(self.K,self.d,self.d))
        self.n = np.zeros((self.K))
        
    def fit(self):
    

        def _eStep():
            for i in range(self.N):
                denom = 0
                for k in range(self.K):
                    denom += self.pi[k] * multivariate_normal.pdf(self.X[i],self.mu[k],self.sgm[k])
                    
                for j in range(self.K):
                    numer = self.pi[j] * multivariate_normal.pdf(self.X[i],self.mu[j],self.sgm[j])
                    self.phi[i][j] = numer/denom                     
        
        def _mStep():
            
            self.n = np.sum(self.phi,axis=0)
            
            for j in range(self.K):
                nj = self.n[j]
                
                self.mu[j] = (1/nj) * np.sum(np.multiply(self.phi[:,j].reshape(self.N,1),self.X),axis=0)
                
                X_mu = self.X - self.mu[j]
                self.sgm = (1/nj) * np.sum(np.multiply(self.phi[:,j].reshape(self.N,1),X_mu.dot(X_mu.T)),axis=0) 
                
                self.pi[j] = nj/np.sum(nj)
                
        def _logLikelihood():
            log_likelihood = 0
            for j in range(self.K):
                log_likelihood += np.sum(multivariate_normal.logpdf(self.X,self.mu[j],self.sgm[j]))
            
            return log_likelihood
            
        for t in range(self.n_iter):
            print t
            _eStep()
            _mStep()
            
            self.logLikelihoods.append(_logLikelihood())
        
        
    