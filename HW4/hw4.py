
# coding: utf-8

# In[ ]:




# In[4]:

import numpy as np
from GMM import VI_GMM
from scipy.special import digamma

X = np.loadtxt('data.txt',delimiter=',')

k=25
model = VI_GMM(X,K=k,n_iter=100)
model.fit()

cluster = [np.argmax(phi) for phi in model.phi]
print cluster