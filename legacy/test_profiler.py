#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:47:34 2019

@author: thomas
"""


import numpy as np

N = 10000
a = np.arange(0, N, dtype=int)
b = np.zeros(N)

def sq(x):
    return x**2

def sq2(x):
    return x**2

def func():
    for i in a:
        b[i] = sq(i)
    print(b)

def func2():
    b = sq2(a)
    print(b)
    
func()
func2()