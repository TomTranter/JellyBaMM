# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:35:17 2020

@author: Tom
"""
import ecm
import configparser
import os
import numpy as np
from scipy.interpolate import NearestNDInterpolator


def load_config(path=None):
    if path is None:
        path = os.getcwd()
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'config.txt'))
    return config


def load_test_config():
    path = os.path.join(ecm.FIXTURES_DIR, 'model')
    path = os.path.join(path, 'example_case_A')
    config = configparser.ConfigParser()
    config_fp = os.path.join(path, 'config.txt')
    config.read(config_fp)
    print('Config file', config_fp, 'loaded')
    print_config(config)
    return config


def print_config(config):
    for sec in config.sections():
        print('=' * 67)
        print(sec)
        print('=' * 67)
        for key in config[sec]:
            print('!', key.ljust(30, ' '), '!',
                  config.get(sec, key).ljust(30, ' '), '!')
            print('-' * 67)


def interpolate_timeseries(project, data):
    im_soft = np.load(os.path.join(ecm.INPUT_DIR, 'im_soft.npz'))['arr_0']
    x_len, y_len = im_soft.shape
    net = project.network
    res_Ts = net.throats('spm_resistor')
    sorted_res_Ts = net['throat.spm_resistor_order'][res_Ts].argsort()
    res_pores = net['pore.coords'][net['throat.conns'][res_Ts[sorted_res_Ts]]]
    res_Ts_coords = np.mean(res_pores, axis=1)
    x = res_Ts_coords[:, 0]
    y = res_Ts_coords[:, 1]
    all_x = []
    all_y = []
    all_t = []
    all_data = []
    for t in range(data.shape[0]):
        all_x = all_x + x.tolist()
        all_y = all_y + y.tolist()
        all_t = all_t + (np.ones(len(x)) * t).tolist()
        all_data = all_data + data[t, :].tolist()
    all_x = np.asarray(all_x)
    all_y = np.asarray(all_y)
    all_t = np.asarray(all_t)
    all_data = np.asarray(all_data)
    points = np.vstack((all_x, all_y, all_t)).T
    myInterpolator = NearestNDInterpolator(points, all_data)
    return myInterpolator
