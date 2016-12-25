#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:39:46 2016
@author: lfawaz
"""
import numpy as np
from scipy.stats import multivariate_normal, wishart
from scipy.special import digamma, gammaln

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
        self.alpha_0 = np.full((self.K),1)
        
        
        self.alpha = self.alpha_0
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
            '''
            This objective function is WRONG!
            '''
            X, d, K = self.X, self.d, self.K
            c,a_0, B_0 = self.c,  self.a_0, self.B_0
            alpha_0, alpha, mu, sgm, a, B, = self.alpha_0, self.alpha, self.mu, self.sgm, self.a, self.B
            phi = self.phi
            
            
            def expLogWish(a, B): 
                sign, logdet = np.linalg.slogdet(B)
                total = 0
                for k in range(d):
                    total += digamma(a+1-j)/2 
                return total + B
           
    
            alpha_gamma = (gammaln(np.sum(alpha)) - gammaln(np.sum(alpha)))
            alpha_0_gamma = -(gammaln(np.sum(alpha_0)) - gammaln(np.sum(alpha_0)))
             
            alpha_all = alpha_gamma + alpha_0_gamma + (alpha_gamma * alpha_0_gamma)
    
            j4 = 0
            j5 = 0
            j7 = 0
            j8 = 0
            j9 = 0
            j10 = 0
            j11 = 0
            j13 = 0
            j14 = 0
            
            for j in range(K):
                sign, logdet_sgm = np.linalg.slogdet(sgm[j])
                sign, logdet_B = np.linalg.slogdet(B[j])
                
                j4 += logdet_sgm * d/5.
                j5 += np.sum((np.trace(sgm[j]) - np.outer(mu[j],mu[j]))/(2.*c))
                X_mu = X - mu[j]
                j7 -= np.sum((np.diag(X_mu.dot(np.linalg.inv(B[j])).dot(X_mu.T))) * phi[:,j])
                phij = np.sum(phi[:,j])
                j8 += np.sum((np.trace(np.linalg.inv(B[j]) + sgm[j])) * (phij/2.))
                j9 += np.sum((expLogWish(a[j], B[j])) * (phij/2.))
                j10 += np.sum((a_0 - a[j])/2. * expLogWish(a[0],B[0]))
                j11 += (np.trace((B_0 + B[j]) * np.linalg.inv(B[j])))/2.
                j13 += (a[j]/2.) * logdet_B
                for k in range(d):
                    j14 += (gammaln((a_0 + 1 - k)/2.) - gammaln((a[j] + 1 - k)/2.))
            
                
                
            alpha_term = np.sum(digamma(alpha_0) - np.sum(digamma(alpha_0)) - (digamma(alpha) - np.sum(digamma(alpha))),axis=0)
            part_b_final = np.sum(np.multiply(phi.T,alpha_term))
            sign_0, logdet_0 = np.linalg.slogdet(B_0) 
            part_final = (a_0/2.) * logdet_0 * K
                
    
            
            
            return np.sum(alpha_all + j4 + j5 + j7 + j8 + j9 + j10 + j11 + j13 + j4 + part_b_final + part_final)
            
              
            
            
        for t in range(self.n_iter):
            if(t%10==0):
                print t
            _update_c()
            _update_n()
            _update_alpha()
            _update_muj()
            _update_lambda()
            self.varObjective.append(_varObjective())
            
           # self.varObjective.append(_varObjective())

           

class GS_GMM ():
    def __init__ (self,X,n_iter=500, c =1/10.,alpha=1):
        self.X = np.asarray(X)
        self.n_iter = n_iter
        self.X_shape = np.shape(self.X)
        self.N = self.X_shape[0]
        self.D = self.X_shape[1]        
        
        
        
        #set Gibbs Sampler parameters
        A = np.cov(self.X.T)
        self.c = c
        self.a = self.D
        self.B = self.c * self.D * A
        self.m = np.mean(self.X,axis=0)
        self.alpha = alpha
        self.lmbda = wishart.rvs(self.a,self.B)
        self.n = np.zeros(self.N)
        self.clusters = np.unique(self.n)
        
        self.mu = np.zeros((1,self.D))
        self.sgm = np.zeros((1,self.D,self.D))
        self.sgm[0] = np.linalg.inv(self.c * self.lmbda) 
        self.mu[0] = multivariate_normal.rvs(self.m,self.sgm[0])
        self.clusters_iter = []
        self.ns_iter = np.zeros((self.n_iter,self.N))
        
    def fit(self):
        
        c , N , X, a, B = self.c, self.N, self.X, self.a, self.B, 
        D, alpha = self.D, self.alpha
        
        def _update_ci(j):
            indices = np.where(self.n == j)
            S = X[indices]
            s = len(S)
            xbar = np.mean(S,axis=0)
            
            def update_sgm():
                aj = a + s
                S_m = S - xbar
                m_S = xbar - self.m
                Bj = B + S_m.T.dot(S_m) + (s/float(s + 1) * m_S.T.dot(m_S))
                lmbdaj = wishart.rvs(aj,Bj)
                cj = s + c         
                self.sgm[j] = np.linalg.inv(cj * lmbdaj)
                
            def update_mu():
                mj = ((c/float(s + c)) * (self.m)) + ((1/float(s + c)) * np.sum(S,axis=0))
                
                self.mu[j] = multivariate_normal.rvs(mj,self.sgm[j])        
        
            update_sgm()
            update_mu()
                
        def _calculateExistingClustersPhiVector(xi,j,minus_i):
            
            
            njMinusI = len(np.argwhere(self.n == j)) - minus_i
            
            phiij = multivariate_normal.pdf(xi,self.mu[j], self.sgm[j]) * njMinusI / float(alpha + N - 1)
            
            
            return phiij
            
        def _calcualteNewClusterPhiValue(xi,j):
            part1 = alpha/float((alpha + N - 1))
            part2 = (c/(np.pi * (1 + c)))
            x_m = xi - self.m
            part3_numr = np.linalg.det(B + (c/(1 + c)) * (x_m).T.dot(x_m)) ** ((-a + 1)/2)
            part3_denom = np.linalg.det(B) ** (-a/2)
            
            part3 = part3_numr/part3_denom
            
            
            lnpart4 = 0
            for d in range(D):
                lnpart4 += gammaln((a + 2 - d)/2.) - gammaln((a + 1 - d)/2.)
            
            part4 = np.exp(lnpart4)
           
            #print part1 * part2 * part3 * part4
            return part1 * part2 * part3 * part4
            
        for t in range(self.n_iter):
            
                
            if(t%10==0):
                print t
                #print self.mu
            #print self.clusters
            for i in range(N):
                
                #calculate phij for all clusters
                phi_i = []
                for j in (self.clusters):
                    
                    minus_i = 0
                    if j == self.n[i]:
                        minus_i = -1
                        
                    phi_i.append(_calculateExistingClustersPhiVector(X[i],j,minus_i))
                    
                
                phi_i.append(_calcualteNewClusterPhiValue(X[i],j))
                
                phi_i = phi_i/np.sum(phi_i)
                
       
                assigned_cluster_index = np.argmax(phi_i)
                
                     
                
                if (len(phi_i) - 1 == assigned_cluster_index):
                    #print "adding new cluster",assigned_cluster
                    self.mu = np.append(self.mu, self.mu[0].reshape(1,D),axis=0)
                    self.sgm = np.append(self.sgm, self.sgm[0].reshape(1,D,D),axis=0)
                    _update_ci(j)
                    self.n[i] = max(self.clusters) + 1
                else:
                    self.n[i] = j
                
                self.clusters = np.unique(self.n)
                
            self.clusters_iter.append(self.clusters)
            self.ns_iter[t] = self.n
            
            
            for cluster in self.clusters:
                #print  cluster
                _update_ci(cluster)
                
            #print "clusters:",self.clusters
            print "cluster count:", len(self.clusters)
            
            
        