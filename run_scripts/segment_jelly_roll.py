# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:31:07 2020

@author: Tom
"""

import jellybamm
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage
import openpnm as op

print('ECM version', jellybamm.__version__)
print('skimage version', skimage.__version__)
wrk = op.Workspace()
print(wrk)

plt.close('all')

os.listdir(jellybamm.INPUT_DIR)

# Return the average of the input images, the mid-half-span and a distance
# transform of the image with centre = 0 to act as radial map
im, mhs, dt = jellybamm.average_images()

plt.figure()
plt.imshow(im)

step = 1  # step to take when binning for averaging
deg = 6  # degree of polynomial fitting when fitting intensity profile
im_soft = jellybamm.remove_beam_hardening(im, dt, step, deg)

# Label Layers
im_soft, cc_im = jellybamm.label_layers(im_soft, dt, mhs,
                                  can_width=30,
                                  im_thresh=19000,
                                  small_feature_size=20000)
np.sum(im_soft[~np.isnan(im_soft)])
np.sum(~np.isnan(im_soft))
np.savez(os.path.join(jellybamm.INPUT_DIR, 'im_soft'), im_soft)
np.savez(os.path.join(jellybamm.INPUT_DIR, 'cc_im'), cc_im)
# Make the spider web network
net = jellybamm.spider_web_network(im_soft, mhs, cc_im, dtheta=10)
jellybamm.plot_topology(net)
prj = net.project
im_spm_map = jellybamm.interpolate_spm_number(prj)
np.savez(os.path.join(jellybamm.INPUT_DIR, 'im_spm_map'), im_spm_map)
