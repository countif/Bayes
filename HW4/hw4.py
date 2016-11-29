#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:53:14 2016

@author: lfawaz
"""

import numpy as np
from GMM import EM_GMM

X = np.loadtxt('data.txt',delimiter=',')

model = EM_GMM(X)

model.fit()