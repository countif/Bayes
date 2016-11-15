
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from VI import VI
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

X_set1 = pd.read_csv("./data_csv/X_set1.csv")
X_set2 = pd.read_csv("./data_csv/X_set2.csv")
X_set3 = pd.read_csv("./data_csv/X_set3.csv")
y_set1 = pd.read_csv("./data_csv/y_set1.csv")
y_set2 = pd.read_csv("./data_csv/y_set2.csv")
y_set3 = pd.read_csv("./data_csv/y_set3.csv")
z_set1 = pd.read_csv("./data_csv/z_set1.csv")
z_set2 = pd.read_csv("./data_csv/z_set2.csv")
z_set3 = pd.read_csv("./data_csv/z_set3.csv")

X = np.asarray(X_set3)
y = np.asarray(y_set3)
z = np.asarray(z_set3)

model = VI(X,y,n_iter=500)
model.iterate()
a,b,e,f, mu, sgm = model.a, model.b,model.e,model.f, model.mu,model.sgm

pd.Series(model.ViLikelihoods).plot(kind="line")


# In[2]:

pd.Series(1/(a/b)).plot(kind="line")


# In[3]:

print f/e


# In[4]:

y_pred = X.dot(mu)
z = np.asarray(z)
z_sinc = np.apply_along_axis(np.sinc,1,z) * 10
plt.plot(z,y_pred)
plt.plot(z,y,'bo-')
plt.plot(z,z_sinc)


# In[ ]:



