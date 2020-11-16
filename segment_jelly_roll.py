# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:31:07 2020

@author: Tom
"""
import ecm 
import matplotlib.pyplot as plt
import numpy as np
import os

plt.close('all')
# Return the average of the input images, the mid-half-span and a distance
# transform of the image with centre = 0 to act as radial map 
im, mhs, dt = ecm.average_images()

plt.figure()
plt.imshow(dt)

step = 1 # step to take when binning for averaging
deg = 6 # degree of polynomial fitting when fitting intensity profile
im_soft = ecm.remove_beam_hardening(im, dt, step, deg)

# Label Layers
im_soft, cc_im = ecm.label_layers(im_soft, dt, mhs,
                             can_width=30,
                             im_thresh=19000,
                             small_feature_size=20000)
np.savez(os.path.join(ecm.INPUT_DIR, 'im_soft'), im_soft)
np.savez(os.path.join(ecm.INPUT_DIR, 'cc_im'), cc_im)
# Make the spider web network
net = ecm.spider_web_network(im_soft, mhs, cc_im, dtheta=10)


