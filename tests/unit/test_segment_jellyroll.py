# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:59:48 2020

@author: tom
"""

import ecm
import matplotlib.pyplot as plt
import os


def test_segment_jellyroll():
    # Return the average of the input images, the mid-half-span and a distance
    # transform of the image with centre = 0 to act as radial map
    im, mhs, dt = ecm.average_images()
    step = 1  # step to take when binning for averaging
    deg = 6  # degree of polynomial fitting when fitting intensity profile
    im_soft = ecm.remove_beam_hardening(im, dt, step, deg)

    # Label Layers
    im_soft, cc_im = ecm.label_layers(im_soft, dt, mhs,
                                      can_width=30,
                                      im_thresh=19000,
                                      small_feature_size=20000)
    # Make the spider web network
    filename = '_test_network.pnm'
    net = ecm.spider_web_network(im_soft, mhs, cc_im, dtheta=20,
                                 path=ecm.INPUT_DIR,
                                 filename=filename)
    prj = net.project
    # make smp map
    ecm.interpolate_spm_number(prj)
    os.remove(os.path.join(ecm.INPUT_DIR, filename))
    plt.close('all')


if __name__ == '__main__':
    test_segment_jellyroll()

