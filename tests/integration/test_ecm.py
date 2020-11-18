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


class ecmTest:
    def _ecm_general(self, config_location):
        wrk = op.Workspace()
        wrk.clear()
        config = configparser.ConfigParser()
        config.read(os.path.join(config_location, 'config.txt'))
        config.set('OUTPUT', 'save', "False")
        config.set('OUTPUT', 'plot', "False")
        config.set('OUTPUT', 'animate', "False")
        I_apps = [config.get('RUN', key) for key in config['RUN'] if 'i_app' in key]
        for I_app in I_apps:
            save_path = config_location + '\\' + I_app + 'A'
            prj, vrs, sols = ecm.run_simulation(float(I_app), save_path, config)
        plt.close('all')

    def test_ecm_spiral(self):
        config_location = os.path.join(ecm.OUTPUT_DIR, 'spiral')
        self._ecm_general(config_location)

    def test_ecm_tomo(self):
        config_location = os.path.join(ecm.OUTPUT_DIR, 'tomography')
        self._ecm_general(config_location)

    def test_ecm_1d(self):
        config_location = os.path.join(ecm.OUTPUT_DIR, '1d')
        self._ecm_general(config_location)


if __name__ == "__main__":
    t = ecmTest()
    self = t
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: ' + item)
            t.__getattribute__(item)()
