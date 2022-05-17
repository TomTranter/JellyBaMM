#
# Utilities
#

import ecm
import os
import numpy as np
from scipy.interpolate import NearestNDInterpolator


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
