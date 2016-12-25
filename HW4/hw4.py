
# coding: utf-8

# In[ ]:




# In[4]:

import numpy as np
from GMM import GS_GMM

X = np.loadtxt('data.txt',delimiter=',')

model_EM_GMM = EM_GMM(X,k=2,n_iter=10)
model_VI_GMM = VI_GMM(X,k=2,n_iter=10)
model_GS_GMM = GS_GMM(X,n_iter=500)

model_EM_GMM.fit()
model_VI_GMM.fit()
model_GS_GMM.fit()