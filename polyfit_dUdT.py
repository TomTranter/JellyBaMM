# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:56:12 2020

@author: Tom
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
#neg dUdT
plt.close('all')

def stretch(data):
    xdata = data[:, 0].copy()
    xdata = xdata - xdata.min()
    xdata = xdata/xdata.max()
    data[:, 0] = xdata
    return data

ndata = np.array([
[0.00025588, 0.00032378],
[0.052556076, 0.000298374],
[0.107537565, 0.000271951],
[0.161220601, 0.000230285],
[0.215393619, 5.69E-06],
[0.270356054, -1.36E-05],
[0.323960148, -2.58E-05],
[0.379330902, -0.000197561],
[0.432951328, -0.000215854],
[0.487802156, -0.000193496],
[0.5409326, -2.89E-05],
[0.593219186, -4.92E-05],
[0.65083297, -5.83E-05],
[0.704426176, -6.65E-05],
[0.758022104, -7.56E-05],
[0.810303245, -9.39E-05],
[0.865254791, -0.000109146],
#[1.0, -0.000109146]
])
ndata = stretch(ndata)
cn = np.polynomial.polynomial.polyfit(ndata[:, 0], ndata[:, 1], 7)
plt.figure()
plt.scatter(ndata[:, 0], ndata[:, 1])
x = np.linspace(0, 1, 101)
def Npoly(sto):
    c = [ 3.24368864e-04, -8.26793440e-04,  1.29426537e-02, -1.17044127e-01,
        3.57057169e-01, -4.90193844e-01,  3.14241304e-01, -7.66056864e-02]
    return c[0] + c[1]*sto + c[2]*sto**2 + c[3]*sto**3 + c[4]*sto**4 + c[5]*sto**5 + c[6]*sto**6 + c[7]*sto**7
plt.plot(x, Npoly(x))

fn = interpolate.CubicSpline(
                ndata[:, 0], ndata[:, 1], extrapolate=False
            )
plt.plot(x, fn(x))

#pos dUdT
pdata = np.array([
#[0.0, 9.81E-06],
[0.216666667, 9.81E-06],
[0.280952381, 3.72E-05],
[0.33452381, 9.51E-06],
[0.388095238, 2.16E-05],
[0.44047619, 1.23E-05],
[0.494047619, -8.21E-06],
[0.547619048, 2.86E-06],
[0.601190476, 5.78E-06],
[0.654761905, -2.70E-05],
[0.707142857, -3.52E-05],
[0.761904762, -3.23E-05],
[0.814285714, -3.04E-05],
[0.867857143, -1.94E-05],
[0.921428571, -4.09E-05],
[0.982142857, -7.36E-05],
])
pdata = stretch(pdata)
cp = np.polynomial.polynomial.polyfit(pdata[:, 0], pdata[:, 1], 5)
ppoly = np.poly1d(np.polyfit(pdata[:, 0], pdata[:, 1], 5))
plt.figure()
plt.scatter(pdata[:, 0], pdata[:, 1])
fp = interpolate.CubicSpline(
                pdata[:, 0], pdata[:, 1], extrapolate=False
            )

#x = np.linspace(pdata[0, 0], pdata[-1, 0], 101)
def Ppoly(sto):
#    c = [-2.48497618e-01,  8.59315741e-01, -1.13830746e+00,  7.09962540e-01,
#         -2.02055921e-01,  2.02525788e-02, -1.10405547e-03,  3.25182032e-04]
    c = [ 1.50623647e-05,  6.18608206e-05,  1.66605563e-04, -2.37881128e-03,
        4.18894188e-03, -2.12586430e-03]
    return c[0] + c[1]*sto + c[2]*sto**2 + c[3]*sto**3 + c[4]*sto**4 + c[5]*sto**5
plt.plot(x, Ppoly(x))
plt.plot(x, fp(x))