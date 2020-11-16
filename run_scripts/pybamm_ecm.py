#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:14:46 2019

@author: tom
.............."""

import pybamm
import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os
import sys

plt.close("all")

#pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
#    save_root = 'C:\\Code\\pybamm_pnm_case4_Chen2020c'
    save_root = sys.argv[-2]
    suffix = sys.argv[-1]
#    save_root = os.getcwd()
    print(save_root, suffix)
    for file in os.listdir(save_root):
        if suffix in file:
            print(file)
            child_root = os.path.join(save_root, file)
            config = configparser.ConfigParser()
            config.read(os.path.join(child_root, 'config.txt'))
            print(ecm.lump_thermal_props(config))
            for sec in config.sections():
                print('='*67)
                print(sec)
                print('='*67)
                for key in config[sec]:
                    print('!', key.ljust(30, ' '), '!', config.get(sec, key).ljust(30, ' '), '!')
                    print('-'*67)
                
            I_apps = [config.get('RUN', key) for key in config['RUN'] if 'i_app' in key]
            for I_app in I_apps:
                save_path = child_root + '\\' + I_app + 'A'
                prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
#                print(save_path)
