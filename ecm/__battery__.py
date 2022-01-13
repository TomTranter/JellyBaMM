#
# Battery related functions
#

import pybamm
import numpy as np


def RKn_fit(x, U0, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    A = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    R = 8.314
    T = 298.15
    F = 96485
    term1 = R * T / F * pybamm.log((1 - x) / x)
    term2 = 0
    for k in range(len(A)):
        a = (2 * x - 1) ** (k + 1)
        b = 2 * x * k * (1 - x)
        c = (2 * x - 1) ** (1 - k)
        term2 += (A[k] / F) * (a - b / c)
    return U0 + term1 + term2


def neg_OCP(sto):
    neg_popt = np.array(
        [
            2.79099024e-01,
            2.72347515e04,
            3.84107939e04,
            2.82700416e04,
            -5.08764455e03,
            5.83084069e04,
            2.74900945e05,
            -1.58889236e05,
            -5.48361415e05,
            3.09910938e05,
            5.56788274e05,
        ]
    )
    return RKn_fit(sto, *neg_popt)


def pos_OCP(sto):
    c = [
        5.88523041,
        -16.64427726,
        65.89481612,
        -131.99750794,
        124.80902818,
        -44.56278259,
    ]
    return (
        c[0]
        + c[1] * sto
        + c[2] * sto ** 2
        + c[3] * sto ** 3
        + c[4] * sto ** 4
        + c[5] * sto ** 5
    )


def neg_dUdT(sto):
    c = [
        3.25182032e-04,
        -1.10405547e-03,
        2.02525788e-02,
        -2.02055921e-01,
        7.09962540e-01,
        -1.13830746e00,
        8.59315741e-01,
        -2.48497618e-01,
    ]
    return (
        c[0]
        + c[1] * sto
        + c[2] * sto ** 2
        + c[3] * sto ** 3
        + c[4] * sto ** 4
        + c[5] * sto ** 5
        + c[6] * sto ** 6
        + c[7] * sto ** 7
    )


def pos_dUdT(sto):
    c = [
        9.90601449e-06,
        -4.77219388e-04,
        4.51317690e-03,
        -1.33763466e-02,
        1.55768635e-02,
        -6.33314715e-03,
    ]
    return (
        c[0]
        + c[1] * sto
        + c[2] * sto ** 2
        + c[3] * sto ** 3
        + c[4] * sto ** 4
        + c[5] * sto ** 5
    )
