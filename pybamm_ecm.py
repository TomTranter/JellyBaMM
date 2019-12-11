#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:14:46 2019

@author: tom
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import ecm


plt.close("all")
# set logging level
pybamm.set_logging_level("INFO")


if __name__ == '__main__':
    I_app = 1.0
    solution = None
    current_state = np.array([])
    I_app = 1.0
    spm_sim = ecm.make_spm()
    spm_sim, R, V_ecm = ecm.step_spm((spm_sim, I_app, 1e-12))
    print(R)
    Nunit = 10
    net, alg, phase, spm_models = ecm.make_net(spm_sim, Nunit, R)
    (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net, alg, V_ecm)
    print('*'*30)
    print('V local pnm', V_local_pnm, '[V]')
    print('I local pnm', I_local_pnm, '[A]')
    print('R local pnm', R_local_pnm, '[Ohm]')
    dt = 2e-3
    Nsteps = 50
    res_Ts = net.throats('spm_resistor')
    terminal_voltages = np.zeros(Nsteps)
    V_test = V_ecm
    tol = 1e-6
    local_R = np.zeros([Nunit, Nsteps])
    stop_R = np.zeros(Nunit, dtype=bool)
    for outer_step in range(Nsteps):
        # Find terminal voltage that satisfy ecm total currents for R
        current_match = False
        max_inner_steps = 100
        inner_step = 0
        damping = 5
        while (inner_step < max_inner_steps) and (not current_match):
            (V_local_pnm, I_local_pnm, R_local_pnm) = ecm.run_ecm(net,
                                                                  alg,
                                                                  V_test)
            tot_I_local_pnm = np.sum(I_local_pnm)
            diff = (I_app - tot_I_local_pnm)/I_app
            if np.absolute(diff) < tol:
                current_match = True
            else:
                V_test *= (1+(diff/damping))
            inner_step += 1
            print('Inner', inner_step, diff, V_test)
        terminal_voltages[outer_step] = V_test
        # I_local_pnm should now match the total applied current
        # Run the spms for the the new I_locals
        data = ecm.pool_spm(zip(spm_models, I_local_pnm, np.ones(Nunit)*dt))
        data = np.asarray(data)
        spm_models = data[:, 0].tolist()
        temp_R = data[:, 1].astype(float)
        sig = 1/temp_R
        sig[temp_R > 1] = 1e-6
        sig[np.isnan(temp_R)] = 1e-6
        phase['throat.electrical_conductance'][res_Ts] = sig
        local_R[:, outer_step] = temp_R
        print(temp_R)
#        for i in range(len(spm_models)):
#            if not stop_R[i]:
##                sim = spm_models[i]
##                temp_R, temp_V_ecm = ecm.step_spm(sim, dt, I_local_pnm[i])
#                if temp_R > 1 or np.isnan(temp_R):
#                    stop_R[i] = True
#                    phase['throat.electrical_conductance'][res_Ts[i]] = 1e-6
#                else:
#                    # Update conductance
#                    local_R[i, outer_step] = temp_R
#                    print(i, temp_R)
#                    sig = 1/temp_R
#                    phase['throat.electrical_conductance'][res_Ts[i]] = sig
#            else:
#                print(i, 'Stopped')
        print('Outer', outer_step)

    fig, ax = plt.subplots()
    ax.plot(terminal_voltages)
    fig, ax = plt.subplots()
    for i in range(Nunit):
        ax.plot(local_R[i, :])

