# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:25:15 2020

@author: Tom
"""

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
#    save_root = sys.argv[-1]
#    save_root = os.getcwd()
    save_root = 'D:\\pybamm_pnm_results\\1d'
    print(save_root)
    config = configparser.ConfigParser()
    config.read(os.path.join(save_root, 'config.txt'))
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
        save_path = save_root + '\\' + I_app + 'A'

        prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
