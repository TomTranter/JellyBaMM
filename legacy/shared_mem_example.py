# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:33:36 2020

@author: Tom
"""

import numpy as np
from npshmex import ProcessPoolExecutor

def add_one(x):
    return x + 1

ex = ProcessPoolExecutor()
big_data = np.ones(int(2e7))

f = ex.submit(add_one, big_data)
print(f.result()[0])