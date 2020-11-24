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


def _teardown(fpath):
    # Delete Data files
    fp = [os.path.join(fpath, file) for file in os.listdir(fpath) if 'A' in file]
    for folder in fp:
        shutil.rmtree(folder)


def _ecm_general(config_location):
    wrk = op.Workspace()
    wrk.clear()
    config = configparser.ConfigParser()
    config.read(os.path.join(config_location, 'config.txt'))
    config.set('OUTPUT', 'save', "True")
    config.set('OUTPUT', 'plot', "False")
    I_apps = [config.get('RUN', key) for key in config['RUN'] if 'i_app' in key]
    for I_app in I_apps:
        save_path = config_location + '\\' + I_app + 'A'
        prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
    plt.close('all')
    # _teardown(config_location)
    assert 1 == 1


def test_ecm_spiral():
    config_location = os.path.join(ecm.OUTPUT_DIR, 'spiral')
    _ecm_general(config_location)


def test_ecm_spiral_tesla():
    config_location = os.path.join(ecm.OUTPUT_DIR, 'spiral_tesla')
    _ecm_general(config_location)


def test_ecm_tomo():
    config_location = os.path.join(ecm.OUTPUT_DIR, 'tomography')
    _ecm_general(config_location)


def test_ecm_1d():
    config_location = os.path.join(ecm.OUTPUT_DIR, '1d')
    _ecm_general(config_location)


def test_func():
    test_config = ecm.load_test_config()
    ecm.print_config(test_config)
