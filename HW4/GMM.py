#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:39:46 2016
@author: lfawaz
"""
import numpy as np
from scipy.stats import multivariate_normal, wishart
from scipy.special import digamma

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
        self.pi = np.absolute(np.random.normal(size=(self.K)))
        self.mu = np.zeros((self.K,self.d))
        self.sgm = np.zeros((self.K,self.d,self.d))
        self.random_matrix = np.random.normal(size=(self.d,self.N))
        
        for i in range(self.K):
            self.sgm[i] = np.identity(self.d)
            index = np.random.choice(np.arange(0,self.N))
            self.mu[i] = self.X[index]

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
                
                
                X_mu_phi = np.multiply(self.phi[:,j].reshape(self.N,1),(self.X - self.mu[j])) 
                self.sgm[j] = (1/nj) * X_mu_phi.T.dot(X_mu_phi) 
                
                self.pi[j] = nj/np.sum(self.n)
            
                
                
        def _logLikelihood():
            log_likelihood = 0
            for i in range(self.N):
                index = np.argmax(self.phi[i])
                
                log_likelihood += np.sum(multivariate_normal.logpdf(self.X[i],self.mu[index],self.sgm[index]))
                
            
            return log_likelihood
            
        for t in range(self.n_iter):
            if(t%10==0):
                print t
                #print self.mu
                #print self.pi
            _eStep()
            #print self.n
            _mStep()
            #print self.sgm
            self.logLikelihoods.append(_logLikelihood())
        
        
class VI_GMM ():
    def __init__ (self,X,K=2,n_iter=100,alpha = 1, c =10):
        self.X = np.asarray(X)
        self.n_iter = n_iter
        self.X_shape = np.shape(self.X)
        self.N = self.X_shape[0]
        self.d = self.X_shape[1]        
        self.varObjective = []
        self.K = K
        self.phi = np.zeros((self.N,self.K))
        
        
        self.I = np.identity(self.d)
        self.c = c
        self.mu_0 = np.zeros((1,self.d))
        self.sgm_0 = (self.I*self.c)        
        self.a_0 = self.d
        self.A = np.cov(self.X.T)
        self.B_0 = (self.d/10.) * self.A
        
        
        self.alpha = np.full((self.K),1)
        self.mu = np.zeros((self.K,self.d))
        self.sgm = np.zeros((self.K,self.d,self.d))
        self.a = np.zeros((self.K,1))
        self.B = np.zeros((self.K,self.d,self.d))
        
        for i in range(self.K):
            index = np.random.choice(np.arange(0,self.N))
            self.mu[i] = self.X[index]
            self.sgm[i] = self.sgm_0
            self.a[i] = self.a_0
            self.B[i] = self.B_0
        
        
    def fit(self):
       
        def _update_c():
            X, alpha, mu, sgm, a, B = self.X, self.alpha, self.mu, self.sgm, self.a, self.B
            
            def _t1(j):
                part1 = 0
                for k in range(self.d):
                    part1 += digamma(((1 - k) + a[j])/2)
                    
                sign, logdet = np.linalg.slogdet(B[j])
                
                t1 = np.sum(part1 - logdet)
                
                return t1
                
            
            def _t2(j):
                
                X_mu = X - mu[j]
                t2 = np.diag(X_mu.dot(a[j] * np.linalg.inv(B[j])).dot(X_mu.T))
                return t2
                
            def _t3(j):
                
                t3 = np.trace(a[j] * np.linalg.inv(B[j]).dot(sgm[j]))
                
                return t3
            
            def _t4(j):
                                
                t4 = np.sum(digamma(alpha[j])/digamma(np.sum(alpha)))
                
                return t4
            
            def _t_function(j):
                
                return np.exp((0.5) * (_t1(j) - _t2(j) - _t3(j)) + _t4(j))
            
            phij = []
            for j in range(self.K):
                phij.append(_t_function(j)) 
                    
            for k in range(len(phij)):
                self.phi[:,k] = phij[k]/np.sum(phij,axis=0)
                
        def _update_n():
            
            self.n = np.sum(self.phi,axis=0)
                    
        def _update_alpha():
            
            self.alpha = self.a_0 + self.n
           
        def _update_muj():
            c, I, n, a, B, X, phi = self.c, self.I, self.n, self.a, self.B, self.X , self.phi
                
            def _update_sgm(j):
                
                self.sgm[j] = np.linalg.inv(((I/c) + (n[j] * a[j]) + np.linalg.inv(B[j]))) 
            
            def _update_mu(j):
                
                sgm = self.sgm
                phi_x = np.sum(np.multiply(phi[:,j].reshape(self.N,1),X),axis=0)
                self.mu[j] = sgm[j].dot(a[j] * np.linalg.inv(B[j]).dot(phi_x))
            
            for j in range(self.K):
             
                _update_sgm(j)
                _update_mu(j)
                
        def _update_lambda():
            a_0 , n, B_0, phi, X, mu, sgm = self.a_0, self.n, self.B_0, self.phi, self.X, self.mu, self.sgm
            
            def _update_a(j):
                self.a[j] = a_0 + n[j]
            
            def _update_B(j):
                
                X_mu_phi = np.multiply(phi[:,j].reshape(self.N,1),(X - mu[j])) 
                
                self.B[j] = B_0 + (X_mu_phi.T.dot(X_mu_phi) + sgm[j]) #np.sum(phi,axis=0)[j] *
                
            for j in range(self.K):
                
                _update_a(j)
                _update_B(j)
                
        def _varObjective():
            varObjective = 0
            
            return varObjective
              
            
            
        for t in range(self.n_iter):
            if(t%10==0):
                print t
            _update_c()
            _update_n()
            _update_alpha()
            _update_muj()
            _update_lambda()
            
           # self.varObjective.append(_varObjective())
        
