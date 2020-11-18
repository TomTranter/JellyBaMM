# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:27:07 2019

@author: Tom
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import os
plt.close('all')

cwd = os.getcwd()
input_dir = os.path.join(cwd, 'data')
files = os.listdir(input_dir)
data = {}
interp_ocv = {}
interp_docv = {}
interp_ent = {}

fig, (ax1, ax2) = plt.subplots(2, 1)
for filename in files:
    filepath = os.path.join(input_dir, filename)
    data[filename] = pd.read_csv(filepath).to_numpy()
    cap = data[filename][:, 0]
    x = cap / cap[-1]
    ocv = data[filename][:, 1]
    ent = data[filename][:, 3]
    ax1.plot(x, ocv, label=filename.split('.')[0])
    ax2.plot(x, ent, label=filename.split('.')[0])
    interp_ocv[filename] = CubicSpline(x, ocv)
    interp_docv[filename] = interp_ocv[filename].derivative()
    interp_ent[filename] = CubicSpline(x, ent)

plt.figure()
plt.plot(x, ocv, 'b')
xlin = np.linspace(0, 1, 101)
plt.plot(xlin, interp_ocv[filename](xlin), 'r--')
