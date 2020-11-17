# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:52:24 2020

@author: Tom
"""

import ecm
import os
root = 'D:\\pybamm_pnm_results\\46800'
cases = ecm.get_cases()
amps = ecm.get_amp_cases()
d = ecm.load_all_data()
for key in cases.keys():
    case_path = os.path.join(root, cases[key]['file'])
    for amp in amps:
        amp_path = os.path.join(case_path, str(amp) + 'A')
        print(amp_path)
        vid_file = os.path.join(amp_path, 'current_density_case_' +
                                str(key) + '_amp_' + str(amp))
        ecm.animate_data4(d, key, amp, [0, 1], vid_file)
