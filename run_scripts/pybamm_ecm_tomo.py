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
import liionpack as lp
import pybamm


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

    # Experiment
    I_app = 3.0
    dt = 30
    Nsteps = 12
    hours = dt * Nsteps / 3600
    experiment = pybamm.Experiment(
        [
            f"Discharge at {I_app} A for {hours} hours",
        ],
        period=f"{dt} seconds",
    )
    save_path = save_root + "\\" + str(I_app) + "A"

    # Geometry of spiral
    tomo_pnm = "spider_net.pnm"
    dtheta = 10
    spacing = 195e-6
    length_3d = 0.08
    pos_tabs = [-1]
    neg_tabs = [0]

    # OpenPNM project
    project, arc_edges = ecm.make_tomo_net(tomo_pnm, dtheta, spacing,
                                           length_3d, pos_tabs, neg_tabs)
    # Parameter set
    param = pybamm.ParameterValues("Chen2020")
    # JellyBaMM discretises the spiral using the electrode height for spiral length
    # This parameter set has the longer length set to the Electrode width
    # We want to swap this round
    param['Electrode width [m]'] = length_3d
    # Run simulation
    project, output = ecm.run_simulation_lp(param, experiment, save_path,
                                            project, config)
    lp.plot_output(output)
