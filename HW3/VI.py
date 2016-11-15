#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:40:22 2016

@author: lfawaz
"""

import numpy as np
import math
from scipy.special import digamma, gammaln

class VI():
    def __init__(self,X,y,n_iter=500,a0=10 ** -16,b0=10 ** -16,e0=1,f0=1):

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_iter = n_iter
        self.X_shape = np.shape(self.X)
        self.N = self.X_shape[0]
        self.d = self.X_shape[1]
        self.w = np.zeros((self.d))
        self.ViLikelihoods = []
        self.K = np.unique(self.y)
        self.W = []  
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.e0 = float(e0)
        self.f0 = float(f0)
        
        #Variables 
        self.a = a0 + 0.5
        self.b = np.full((self.d),b0)
        self.e = e0 + self.N/2
        self.f = f0
        self.mu = np.zeros((self.d))
        self.sgm = np.identity((self.d))
    
        
    
        
    def _updateLamdba(self):
        #look at mu.T if values don't work
        f_part1 = (self.y.T  - self.X.dot(self.mu)) **2
        f_part2 = np.diag(self.X.dot(self.sgm).dot(self.X.T))
        f_parts = np.sum(f_part1 + f_part2)
        self.f = self.f0 + (0.5 * f_parts)
        #print self.f
    def _updateAlpha(self):
        mu = self.mu.reshape(self.d,1)
        self.b = self.b0 + (0.5 * np.diag((self.sgm + (mu.dot(mu.T)))))
        #print np.sum(self.b)
        
        
    def _updateW(self):
        self.sgm = np.linalg.inv(np.diag(self.a/self.b) + ((self.e/self.f) * self.X.T.dot(self.X)))
        self.mu = self.sgm.dot((self.e/self.f) *(np.sum(np.multiply(self.y,self.X),axis=0)))

    def _calcViLikelihood(self):
        mu = self.mu.reshape(self.d,1)                           
    
        lambda_part1 = (self.e0 * np.log(self.f0) - gammaln(self.e0))
        lambda_part2 = -(self.e * np.log(self.f) - gammaln(self.e))
        lambda_part3 = (self.e0 - self.e) * (digamma(self.e) - np.log(self.f)) 
        lambda_part4 = - ((self.f0 - self.f) * (self.e / self.f))
        
        lambda_part = lambda_part1 + lambda_part2 + lambda_part3 + lambda_part4   
        #print lambda_part
        
        alpha_part1 = self.d * (self.a0 * np.log(self.b0) - gammaln(self.a0))
        alpha_part2 = - (self.a * np.sum(np.log(self.b)) - (self.d * gammaln(self.a)))
        alpha_part3 = (self.a0 - self.a) * np.sum(digamma(self.a) - np.log(self.b))
        alpha_part4 = - (self.b0*np.sum(self.a/self.b)) + (self.d * self.a)
        
        alpha_part = alpha_part1 + alpha_part2 + alpha_part3 + alpha_part4 
        
        
        
        w_part1 = 0.5 * np.sum(digamma(self.a) - np.log(self.b))
        w_part2 = -0.5 * np.sum(np.diag((self.sgm + (mu.dot(mu.T)))) * (self.a/self.b)) 
        sign, logdet = np.linalg.slogdet(self.sgm)
        w_part3 = 0.5 * (sign * logdet) 
        w_part = w_part1 + w_part2 + w_part3  
        
        
        y_part1 = self.N/2 * (digamma(self.e) - np.log(self.f)) 
        #fix this one if you find an error
        y_part2 = -(self.f/(2*self.e) * np.sum(((self.y - self.X.dot(self.mu)) ** 2 + self.X.dot(self.sgm).dot(self.X.T))))
        
        y_part = y_part1 + y_part2 
    
        return lambda_part + alpha_part + w_part + y_part
    
    def iterate(self):
            for i in range(self.n_iter):
                #print "iter: ",i, "Mu:",np.sum(self.mu), "sigma:",np.sum(self.sgm)
                
               
                self._updateLamdba()
                #print self.f
                self._updateAlpha()
                #print self.b
                self._updateW()
                #print self.sgm
                
                self.ViLikelihoods.append(self._calcViLikelihood())