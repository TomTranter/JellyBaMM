# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:24:19 2020

@author: tom
"""

import ecm
import matplotlib.pyplot as plt
import os


root = ecm.TEST_CASES_DIR
children = []
for file in os.listdir(root):
    child = os.path.join(root, file)
    children.append(child)


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
    plt.close('all')
    assert 1 == 1


def test_multivar_subplot():
    data = ecm.load_cases(root)
    case_index = 0
    case_folder = children[case_index]
    amps = ecm.get_amp_cases(case_folder)
    case = list(data.keys())[case_index]
    ecm.multi_var_subplot(data, [case], amps, [2, 0], landscape=False)
    plt.close('all')
    assert 1 == 1


def test_spacetime():
    data = ecm.load_cases(root)
    amps = ecm.get_amp_cases(children[0])
    cases = list(data.keys())
    ecm.spacetime(data, cases, amps, var=0, group='neg', normed=True)
    plt.close('all')
    assert 1 == 1


def test_chargeogram():
    data = ecm.load_cases(root)
    amps = ecm.get_amp_cases(children[0])
    cases = list(data.keys())
    ecm.chargeogram(data, cases, amps, group='neg')
    plt.close('all')
    assert 1 == 1


def test_animate():
    data = ecm.load_cases(root)
    amps = ecm.get_amp_cases(children[0])
    cases = list(data.keys())
    fname = os.path.join(children[0], 'test.mp4')
    ecm.animate_data4(data, cases[0], amps[0], variables=[0, 1], filename=fname)
    os.remove(fname)


if __name__ == '__main__':
    test_jellyroll_subplot()
    test_multivar_subplot()
    test_spacetime()
    test_chargeogram()
    test_animate()
