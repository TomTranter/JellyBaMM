# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:14:54 2020

@author: Tom
"""

import ecm
import numpy as np
import matplotlib.pyplot as plt

d = ecm.load_all_data()
cases = ecm.get_cases()
# Base Case all Amps - HTC 5 - 2 Tabs
#fig1 = ecm.multi_var_subplot(d, [0], [4, 6, 8, 10], [0, 1])
## Base Case all Amps - Normalized
#fig2 = ecm.multi_var_subplot(d, [0], [4, 6, 8, 10], [0, 1], normed=True)
## All HTC cases - 2 tabs, 10 A
#fig3 = ecm.multi_var_subplot(d, [0, 1, 2, 3], [10], [0, 1])
## All HTC cases - 5 tabs, 10 A
#fig4 = ecm.multi_var_subplot(d, [8, 9, 10, 11], [10], [0, 1])
## 100 HTC cases - 2 tabs, 4, 10 A - neg_cc_econd
#fig5 = ecm.multi_var_subplot(d, [3, 12], [4, 10], [0, 1])
## 100 HTC cases - 2 tabs, 4, 10 A - variable k
#fig6 = ecm.multi_var_subplot(d, [3, 13], [4, 10], [0, 1])
## Base Case all Amps - Average Heating
#fig7 = ecm.multi_var_subplot(d, [0], [4, 6, 8, 10], [7, 1])

ecm.animate_data4(d, 0, 10, [0, 1], 'test_new_ani')