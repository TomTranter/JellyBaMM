# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:24:19 2020

@author: tom
"""

import ecm
import matplotlib.pyplot as plt
import os
import openpnm as op


wrk = op.Workspace()
close_plots = True
root = ecm.TEST_CASES_DIR
children = []
for file in os.listdir(root):
    child = os.path.join(root, file)
    children.append(child)


def _load(filepath):
    data = ecm.load_cases(filepath)
    cases = list(data.keys())
    amps = ecm.get_amp_cases(os.path.join(filepath, cases[0]))
    return data, amps, cases


def test_jellyroll_subplot():
    data = ecm.load_cases(root)
    case_index = 0
    amp_index = 0
    var_index = 0
    case_folder = children[case_index]
    amps = ecm.get_amp_cases(case_folder)
    case = list(data.keys())[case_index]
    ecm.jellyroll_subplot(data, case, amps[amp_index], var=var_index,
                          soc_list=[[1.0, 0.99], [0.98, 0.97]],
                          global_range=False, dp=1)
    if close_plots:
        plt.close('all')
    wrk.clear()
    assert 1 == 1


def test_multivar_subplot():
    data, amps, cases = _load(root)
    ecm.multi_var_subplot(data, [cases[0]], amps, [2, 0], landscape=False)
    if close_plots:
        plt.close('all')
    wrk.clear()
    assert 1 == 1


def test_spacetime():
    data, amps, cases = _load(root)
    ecm.spacetime(data, cases, amps, var=0, group='neg', normed=True)
    if close_plots:
        plt.close('all')
    wrk.clear()
    assert 1 == 1


def test_chargeogram():
    data, amps, cases = _load(root)
    ecm.chargeogram(data, cases, amps, group='neg')
    if close_plots:
        plt.close('all')
    wrk.clear()
    assert 1 == 1


def test_animate():
    data, amps, cases = _load(root)
    fname = os.path.join(children[0], 'test.mp4')
    ecm.animate_data4(data, cases[0], amps[0], variables=[0, 1], filename=fname)
    os.remove(fname)
    wrk.clear()


def test_super_subplot():
    filepath = os.path.join(ecm.FIXTURES_DIR, 'model')
    data, amps, cases = _load(filepath)
    ecm.super_subplot(data, [cases[0]], [cases[0]], amps[0])
    if close_plots:
        plt.close('all')
    wrk.clear()
    assert 1 == 1


if __name__ == '__main__':
    test_jellyroll_subplot()
    test_multivar_subplot()
    test_spacetime()
    test_chargeogram()
    test_animate()
    test_super_subplot()
