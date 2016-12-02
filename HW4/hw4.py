#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:53:14 2016

@author: lfawaz
"""

import numpy as np
from GMM import EM_GMM
import pandas as pd

X = np.loadtxt('data.txt',delimiter=',')

k=10
model = EM_GMM(X,K=k,n_iter=100)
model.fit()
pd.Series(model.logLikelihoods).plot()