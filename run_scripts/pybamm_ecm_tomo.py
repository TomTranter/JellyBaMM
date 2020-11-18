# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:25:15 2020

@author: Tom
"""

import openpnm as op
import matplotlib.pyplot as plt
import ecm
import configparser
import os

plt.close("all")

#pybamm.set_logging_level("INFO")
wrk = op.Workspace()
wrk.clear()


if __name__ == "__main__":
    save_root = os.path.join(ecm.OUTPUT_DIR, 'tomography')
    print(save_root)
    config = configparser.ConfigParser()
    config.read(os.path.join(save_root, 'config.txt'))
    print(ecm.lump_thermal_props(config))
    ecm.print_config(config)

    I_apps = [config.get('RUN', key) for key in config['RUN'] if 'i_app' in key]
    for I_app in I_apps:
        save_path = save_root + '\\' + I_app + 'A'
        prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
