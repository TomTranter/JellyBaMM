#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:10:11 2019

@author: tom
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def u_lico_moura(sto):
    u_eq = (
        2.16216
        + 0.07645 * np.tanh(30.834 - 54.4806 * sto)
        + 2.1581 * np.tanh(52.294 - 50.294 * sto)
        - 0.14169 * np.tanh(11.0923 - 19.8543 * sto)
        + 0.2051 * np.tanh(1.4684 - 5.4888 * sto)
        + 0.2531 * np.tanh((-sto + 0.56478) / 0.1316)
        - 0.02167 * np.tanh((sto - 0.525) / 0.006)
    )
    return u_eq


def u_lico(sto, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s):
    u_eq = (
        a
        + b * np.tanh(c - d * sto)
        + e * np.tanh(f - g * sto)
        + h * np.tanh(i - j * sto)
        + k * np.tanh(l - m * sto)
        + n * np.tanh((-sto + o) / p)
        - q * np.tanh((sto - r) / s)
    )
    return u_eq

xdata = np.linspace(0, 1, 101)
ydata = u_lico_moura(xdata)

popt, pcov = curve_fit(u_lico, xdata, ydata, bounds=(-10, 60), maxfev=1000000)

plt.figure()
plt.plot(xdata, ydata)
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s = popt
plt.plot(xdata, u_lico(xdata, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s))