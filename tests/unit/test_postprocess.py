# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:24:19 2020

@author: tom
"""

import ecm
import openpnm as op
import matplotlib.pyplot as plt
import configparser
import os
import shutil


root = os.path.join(ecm.OUTPUT_DIR, 'cases')
children = []


def _ecm_general(config_location):
    wrk = op.Workspace()
    wrk.clear()
    config = configparser.ConfigParser()
    config.read(os.path.join(config_location, 'config.txt'))
    config.set('OUTPUT', 'save', "True")
    config.set('OUTPUT', 'plot', "False")
    config.set('OUTPUT', 'animate', "False")
    I_apps = [config.get('RUN', key) for key in config['RUN'] if 'i_app' in key]
    for I_app in I_apps:
        save_path = os.path.join(config_location, str(I_app) + 'A')
        prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
    plt.close('all')
    assert 1 == 1


def setup():
    # Generate Data files
    for file in os.listdir(root):
        child = os.path.join(root, file)
        print('processing case', child)
        _ecm_general(child)
        children.append(child)


def teardown():
    # Delete Data files
    for child in children:
        fp = [os.path.join(child, file) for file in os.listdir(child) if 'A' in file]
        for folder in fp:
            shutil.rmtree(folder)


def test_load_data():
    d = ecm.load_data(children[0])
    assert len(d.keys()) > 0


def test_load_cases():
    d = ecm.load_cases(root)
    assert len(d.keys()) > 0
    return d


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
    ecm.chargeogram(d, cases, amps, group='neg')
    plt.close('all')
    assert 1 == 1


if __name__ == '__main__':
    setup()
    test_load_data()
    d = test_load_cases()
    test_jellyroll_subplot()
    test_multivar_subplot()
    test_spacetime()
    test_chargeogram()
    teardown()
