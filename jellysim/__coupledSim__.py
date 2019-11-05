#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:42:06 2019

@author: thomas
"""
import jellysim as js
import time
import numpy as np


class coupledSim(object):
    def __init__(self):
        pass

    def setup(self, options):
        self.options = options.copy()
        pnm = js.pnm_runner()
        if options['domain'] == 'tomography':
            pnm.setup_tomo()
        else:
            pnm.setup_jelly(Nlayers=options['Nlayers'],
                            dtheta=options['dtheta'],
                            spacing=options['spacing'])
        pnm.setup_geometry(dtheta=options['dtheta'],
                           spacing=options['spacing'],
                           length_3d=options['length_3d'])
        pnm.setup_thermal(options['T0'],
                          options['cp'],
                          options['rho'],
                          options['K0'],
                          options['heat_transfer_coefficient'])
        spm = js.spm_runner()

        self.runners = {'pnm': pnm,
                        'spm': spm}
        spm.setup(I_app=options['I_app_mag'],
                  T0=options['T0'],
                  cc_cond_neg=options['cc_cond_neg'],
                  cc_cond_pos=options['cc_cond_pos'],
                  z_edges=pnm.arc_edges.copy(),
                  length_3d=options['length_3d'])

    def run_thermal(self):
        pnm = self.runners['pnm']
        T0 = self.options['T0']
        C_rate = 2.0
        heat_source = np.ones(pnm.Nunit) * 25e3 * C_rate
        time_step = 0.01
        pnm.run_step(heat_source, time_step, BC_value=T0)
        pnm.plot_temperature_profile()

    def run(self, n_steps, time_step, initialize=True):
        pnm = self.runners['pnm']
        spm = self.runners['spm']
        options = self.options
        start_time = time.time()
        if initialize:
            # Initialize - Run through loop to get temperature then discard
            # solution with small time step
            print('*'*30)
            print('Initializing')
            print('*'*30)
            spm.run_step(time_step, n_subs=10)
            heat_source = spm.get_heat_source()
            print("Heat Source", np.mean(heat_source))
            pnm.run_step(heat_source, time_step, BC_value=options['T0'])
            global_temperature = pnm.get_average_temperature()
            print("Global Temperature", np.mean(global_temperature))
            T_diff = global_temperature.max() - global_temperature.min()
            print("Temperature Range", T_diff)
            spm.update_external_temperature(global_temperature)
            spm.solution = None
        print("*" * 30)
        print("Running Steps")
        print("*" * 30)
        termination = False
        i = 0
        while i < n_steps and not termination:
            spm.run_step(time_step, n_subs=10)
            heat_source = spm.get_heat_source()
            print("Heat Source", np.mean(heat_source))
            pnm.run_step(heat_source, time_step, BC_value=options['T0'])
            global_temperature = pnm.get_average_temperature()
            print("Global Temperature", np.mean(global_temperature))
            T_diff = global_temperature.max() - global_temperature.min()
            print("Temperature Range", T_diff)
            spm.update_external_temperature(global_temperature)
            i += 1
            if spm.solution.termination != 'final time':
                termination = True
        end_time = time.time()
        print("*" * 30)
        print("Simulation Time", np.around(end_time - start_time, 2), "s")
        print("*" * 30)

    def plots(self):
        pnm = self.runners['pnm']
        spm = self.runners['spm']
        spm.plot()
        vars = [
            "X-averaged total heating [A.V.m-3]",
            "X-averaged cell temperature [K]",
            "X-averaged positive particle surface concentration [mol.m-3]",
            "X-averaged negative particle surface concentration [mol.m-3]",
            "Negative current collector potential [V]",
            "Positive current collector potential [V]",
            "Local current collector potential difference [V]",
        ]

        tind = -1
        var = vars[2]
        tind = -1
        data = pnm.convert_spm_data(spm.get_processed_variable(var, time_index=tind))
        pnm.plot_pore_data(data, title=var + " @ time " + str(spm.solution.t[tind]))
        pnm.plot_temperature_profile()
        spm.plot_3d()
        var = "X-averaged positive particle surface concentration [mol.m-3]"
        js.utils.plot_time_series(var, pnm, spm)

    def save(self, name):
        for key in self.runners.keys():
            js.save_obj(name+'_'+key, self.runners[key])
