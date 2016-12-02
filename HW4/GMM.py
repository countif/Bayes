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
        self.pi = np.random.normal(size=(self.K))
        self.pi = self.pi/np.sum(self.pi)
        self.mu = np.random.uniform(size=(self.K,self.d))
        self.sgm = np.random.uniform(size=(self.K,self.d,self.d))
        for i in range(self.K):
            self.sgm[i] = self.sgm[i].dot(self.sgm[i].T)

        self.n = np.zeros((self.K))
        
    def fit(self):
    

        def _eStep():
            #print self.sgm
            small = 10 ** -16
            for i in range(self.N):
                denom = 0
                for k in range(self.K):
                    denom += self.pi[k] * multivariate_normal.pdf(self.X[i],self.mu[k],self.sgm[k])
                    
                for j in range(self.K):
                    numer = self.pi[j] * multivariate_normal.pdf(self.X[i],self.mu[j],self.sgm[j])
        
                    self.phi[i][j] = (numer + small)/(denom + small)
                    #print self.phi[i][j],multivariate_normal.pdf(self.X[i],self.mu[j],self.sgm[j])
                    
            #print np.shape(self.sgm)
            
        def _mStep():
            
            self.n = np.sum(self.phi,axis=0)
            
            for j in range(self.K):
                
                nj = self.n[j]
                
                self.mu[j] = (1/nj) * np.sum(np.multiply(self.phi[:,j].reshape(self.N,1),self.X),axis=0)
                
                phij_X_mu = np.multiply(self.phi[:,j].reshape(self.N,1),(self.X - self.mu[j]))
                #print np.shape(self.sgm)
                self.sgm[j] = (1/nj) * phij_X_mu.T.dot(phij_X_mu)
                
                self.pi[j] = nj/np.sum(nj)
                
                
                
        def _logLikelihood():
            log_likelihood = 0
            for j in range(self.K):
                #print self.mu[j]
                #print self.sgm[j]
                log_likelihood += -1*np.sum(multivariate_normal.logpdf(self.X,self.mu[j],self.sgm[j]))
            
            return log_likelihood
            
        for t in range(self.n_iter):
            if(t%10==0):
                print t
            _eStep()
            _mStep()
            
            self.logLikelihoods.append(_logLikelihood())
        print np.shape(self.phi)
        
    