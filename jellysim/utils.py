#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:20:59 2019

@author: thomas
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def specific_cap(diam, height, cap):
    a = np.pi * (diam / 2) ** 2
    v = a * height
    spec = cap / v
    return spec


def save_obj(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


def plot_time_series(var, pnm=None, spm=None):
    all_time_data = spm.get_processed_variable(var, time_index=None)
    data = pnm.convert_spm_data(all_time_data[:, 0])
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(bottom=0.25)
    bulk_Ps = pnm.net.pores("free_stream", mode="not")
    coords = pnm.net["pore.coords"][bulk_Ps]
    xmin = coords[:, 0].min() * 1.05
    ymin = coords[:, 1].min() * 1.05
    xmax = coords[:, 0].max() * 1.05
    ymax = coords[:, 1].max() * 1.05
    mappable = ax.scatter(coords[:, 0], coords[:, 1], c=data[bulk_Ps])
    mappable.set_clim([all_time_data.min(), all_time_data.max()])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(mappable)
    plt.title(var)
    ax1_pos = fig.add_axes([0.2, 0.1, 0.65, 0.03])
    s1 = Slider(ax1_pos, "Time", 0, spm.solution.t.max(), valinit=0)

    def update1(v):
        tind = np.argwhere(spm.solution.t > s1.val).min()
        data = pnm.convert_spm_data(all_time_data[:, tind])
        mappable.set_array(data[bulk_Ps])
        fig.canvas.draw_idle()

    s1.on_changed(update1)
    plt.show()
