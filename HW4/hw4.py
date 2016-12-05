
# coding: utf-8

# In[ ]:




# In[4]:

import numpy as np
from GMM import VI_GMM
from scipy.special import digamma

X = np.loadtxt('data.txt',delimiter=',')

k=10
model = VI_GMM(X,K=k,n_iter=100)
model.fit()

print model.phi
#print "B_0",model.B_0

#d = np.shape(X)[1]
#A = np.cov(X.T) * (d/10.)

#print A

#print np.sum(digamma([1]))