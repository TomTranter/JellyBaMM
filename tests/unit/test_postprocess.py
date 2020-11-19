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


root = os.path.join(ecm.OUTPUT_DIR, 'tomography')


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
        save_path = config_location + '\\' + I_app + 'A'
        prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
    plt.close('all')
    assert 1 == 1


def setup():
    # Generate Data files
    _ecm_general(root)


def teardown():
    # Delete Data files
    fp = [os.path.join(root, file) for file in os.listdir(root) if 'A' in file]
    for folder in fp:
        shutil.rmtree(folder)


def test_load_data():
    d = ecm.load_data(root)
    assert len(d.keys()) > 0
    return d


if __name__ == '__main__':
    setup()
    d = test_load_data()
    teardown()
