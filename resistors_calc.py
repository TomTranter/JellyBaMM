# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:04:02 2020

@author: Tom
"""
import numpy as np
import openpnm as op

n_lic6 = 8
n_lico = 7
n_al = 1
n_cu = 1
n_sep = 2



lens = np.array([n_lic6, n_lico, n_al, n_cu, n_sep])

#a_al = 1.0
#a_lic6 = 1.0
#a_sep = 1.0
#a_lico = 1.0
#a_cu = 1.0


a_lic6 = 5.370092e-7
a_lico = 5.346227e-7
a_al = 9.754474e-5
a_cu = 1.157244e-4
a_sep = 1.675415e-7

alphas = np.array([a_lic6, a_lico, a_al, a_cu, a_sep])

R_lic6 = 1/a_lic6
R_lico = 1/a_lico
R_al = 1/a_al
R_cu = 1/a_cu
R_sep = 1/a_sep

res = np.array([R_lic6, R_lico, R_al, R_cu, R_sep])
print(res)
R_series = np.sum(lens*res)/np.sum(lens)
print('R Series', R_series)
print('Cond Series', 1/R_series)
#R_parallel = 1/(np.sum(lens/res)/np.sum(lens))
R_parallel = np.sum(lens)/np.sum(lens/res)
print('R Parallel', R_parallel)
print('Cond Parallel', 1/R_parallel)

net_series = op.network.Cubic([np.sum(lens)+1, 1, 1])
phase = op.phases.GenericPhase(net_series)
phase['throat.diffusive_conductance'] = 1.0
phase['throat.diffusive_conductance'][:n_al] = a_al
phase['throat.diffusive_conductance'][n_al:n_al+n_lic6] = a_lic6
phase['throat.diffusive_conductance'][n_al+n_lic6:n_al+n_lic6+n_sep] = a_sep
phase['throat.diffusive_conductance'][n_al+n_lic6+n_sep:n_al+n_lic6+n_sep+n_lico] = a_lico
phase['throat.diffusive_conductance'][n_al+n_lic6+n_sep+n_lico:n_al+n_lic6+n_sep+n_lico+n_cu] = a_cu

diffs = phase['throat.diffusive_conductance'].copy()
alg = op.algorithms.FickianDiffusion(network=net_series)
alg.setup(phase=phase)
alg.set_value_BC(pores=net_series.pores('front'), values=1.0)
alg.set_value_BC(pores=net_series.pores('back'), values=0.0)
alg.run()
deff_series = alg.calc_effective_diffusivity(domain_area=1.0, domain_length=np.sum(lens))[0]

net_parallel = op.network.Cubic([2, np.sum(lens), 1])
Ts = net_parallel.find_neighbor_throats(pores=net_parallel.pores('front'), mode='xor')
phase = op.phases.GenericPhase(net_parallel)
phase['throat.diffusive_conductance'] = 1.0
phase['throat.diffusive_conductance'][Ts] = diffs
alg = op.algorithms.FickianDiffusion(network=net_parallel)
alg.setup(phase=phase)
alg.set_value_BC(pores=net_parallel.pores('front'), values=1.0)
alg.set_value_BC(pores=net_parallel.pores('back'), values=0.0)
alg.run()
deff_parallel = alg.calc_effective_diffusivity(domain_area=np.sum(lens), domain_length=1.0)[0]
print('Series', deff_series, 'Parallel', deff_parallel, 'Ratio', deff_parallel/deff_series)