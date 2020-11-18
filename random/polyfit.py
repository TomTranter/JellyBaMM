# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:25:05 2020

@author: Tom
"""
import warnings
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    p30 = np.poly1d(np.polyfit(x, y, 30))

#p = np.poly1d(z)
print(p30(0.5))
