import pandas as pd
import numpy as np
from scipy.special import gammaln
import math

class StudenttNB():
    def __init__(self,X,y,m=0,a=1,b=1,c=1,e=1,f=1,pi=0.5):
        self.m = m
        self.a = a
        self.b = b
        self.c = c
        self.e = e
        self.f = f
        self.pi = pi
        self.X = X
        self.y = y
       

    def fit(self):
        
        #Intialize categories (K), number of records N, dimensions d
        K = set(self.y[0])
        N = len(self.X)
        d = len(self.X.iloc[0])

        #initialize y param
        ycolumns = ['e','f','pi']
        yparams = pd.DataFrame(index=K,columns=ycolumns,dtype=float)
        yparams['e'] = self.e
        yparams['f'] = self.f
        yparams['pi'] = self.pi

        #initlialize x param
        Xcolumns = ['m','a','b','c']
        xparams = pd.DataFrame(np.zeros((len(K),d))).stack()
        Xparams = pd.DataFrame(xparams)
        Xparams.columns = ['m']
        Xparams['m'] = self.m
        Xparams['a'] = self.a
        Xparams['b'] = self.b
        Xparams['c'] = self.c

        def _train_k(yparam, xparam):
            yp = yparam
            xp = xparam

            yk = self.y[self.y[0] == k]
            Xk = self.X.ix[yk.index]

            variable = self.e if k == 1 else self.f
            pi = (variable + len(yk))/float((N + self.e + self.f))
            yp['pi'] = pi
            #train xparam for posterior x (normal gamma)

            mean = Xk.mean()
            sigma = Xk.var()
            n = len(Xk)

            
            m, a, b, c = xp['m'], xp['a'],xp['b'],xp['c']

            post_m = (a * m + n*mean)/(a+n)
            post_a = a + n
            post_b = b + n/2
            post_c = c + ((n*a * (mean - m)**2)/(a+n) + n*sigma)/2.
            


            xp['m'] = post_m
            xp['a'] = post_a
            xp['b'] = post_b
            xp['c'] = post_c


            return yp, xp

        for k in K:
        #train yparam for posterior y 
            y_post, x_post = _train_k(yparams.ix[k],Xparams.ix[k])
           
            Xparams['m'].ix[k] = list(x_post['m'])
            Xparams['a'].ix[k] = list(x_post['a'])
            Xparams['b'].ix[k] = list(x_post['b'])
            Xparams['c'].ix[k] = list(x_post['c'])
            yparams.ix[k] = y_post

        return Xparams, yparams


    def predict_proba(self,Xtest):
        Xparams, yparams = self.fit()
        K = set(self.y[0])
        _student_t = np.zeros((len(Xtest),len(K)))
        predict_proba = np.zeros((len(Xtest),len(K)))
        for i in range(len(Xtest)):
            
            k_pred_proba = []
            for k in K:
                #print k
                prior = yparams['pi'].ix[k]
                likelihood = self._student_t_distribution(Xparams.ix[k],xtest=Xtest.iloc[i]).prod()
                predictive = likelihood * prior
                _student_t[i][k] = predictive
                
            for k in K:
                predict_proba[i][k] = _student_t[i][k]/np.sum(_student_t[i])   
                
        return predict_proba
        
    def _student_t_distribution(self,xparam,xtest):
        xp = xparam
        m, a , b, c =  xp['m'], xp['a'],xp['b'],xp['c']
        b_pred = b + 0.5
        c_pred = (c + (a/2*((xtest - m)**2)/(a + 1)))

        p_student_t =  math.e**(gammaln(b_pred) - gammaln(b)) * (np.sqrt(a)/np.sqrt(math.pi*(a+1)*2)) * np.exp(((b*np.log(c)) - (b_pred*np.log(c_pred)))) 

        return p_student_t

    def predict(self,Xtest):
        predict_proba = self.predict_proba(Xtest)
        predict = []
        for i in range(len(Xtest)):
            predict.append(np.argmax(predict_proba[i]))
             
        return predict


