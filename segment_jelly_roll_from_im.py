# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:33:07 2020

@author: Tom
"""
import numpy as np
import ecm
import matplotlib.pyplot as plt
import os

plt.close('all')

im_soft = np.load(os.path.join(ecm.INPUT_DIR, 'im_soft.npz'))['arr_0']
cc_im = np.load(os.path.join(ecm.INPUT_DIR, 'cc_im.npz'))['arr_0']
# Make the spider web network
mhs = int(cc_im.shape[0] / 2)
net = ecm.spider_web_network(im_soft, mhs, cc_im, dtheta=20)
prj = net.project
im_spm_map = ecm.interpolate_spm_number(prj)
np.savez(os.path.join(ecm.INPUT_DIR, 'im_spm_map'), im_spm_map)
