#!/usr/bin/env python3
# @File    : SpinPlot.py
# @Time    : 9/5/2021 9:24 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import numpy as np


def force_aspect(ax, aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def fit_centralize_heatmap(x_axis, y_axis):
    # minimum at the top left
    if x_axis.shape[0] == 1:
        block_x_width = 0.1
    else:
        block_x_width = (np.max(x_axis) - np.min(x_axis)) / (x_axis.shape[0] - 1)
    if y_axis.shape[0] == 1:
        block_y_width = 0.1
    else:
        block_y_width = (np.max(y_axis) - np.min(y_axis)) / (y_axis.shape[0] - 1)
    return [np.min(x_axis) - block_x_width / 2,
            np.max(x_axis) + block_x_width / 2,
            np.max(y_axis) + block_y_width / 2,
            np.min(y_axis) - block_y_width / 2]


def modify_if_single_tick(ax, x, y):
    if y.shape[0] == 1:
        ax.set_yticks(y)
    if x.shape[0] == 1:
        ax.set_xticks(x)
