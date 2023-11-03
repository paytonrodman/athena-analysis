#!/usr/bin/env python3
#
# plt_beta.py
#
# A program to plot the time-series of the average magnetic plasma beta, from data generated
# by calc_beta.py.
#
# Usage: python plt_beta.py [options]
#
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
import numpy as np
import pandas as pd
import re

# Athena++ modules (require sys.path.insert above)
import AAT

def main(**kwargs):
    dens = []
    # get pre-defined labels and line colours for each simulation
    f = args.file[0]
    slash_list = [m.start() for m in re.finditer('/', f.name)]
    prob_id = f.name[slash_list[-2]+1:slash_list[-1]]

    df = pd.read_csv(f, delimiter='\t', usecols=['sim_time', 'line_density'])
    t = df['sim_time'].to_list()
    d = df['line_density'].to_list()

    time = t
    for di in d:
        di_reduc = np.fromstring(di.strip("[]"), sep=', ')
        dens.append(di_reduc)

    time, dens = zip(*sorted(zip(time, dens)))
    dens = np.asarray(dens)

    r_min = -290
    r_max = 290
    N = 50

    if args.early:
        t_min = 0
        t_max = 1.22e5
    else:
        t_min = 0
        t_max = time[-1]

    tmax_index = AAT.find_nearest(time, t_max)
    t_max = time[tmax_index]
    time = time[:tmax_index]
    dens = dens[:tmax_index]

    zl, zr = np.split(dens, 2, axis=1)
    z_nan = np.nan*np.ones((np.size(time),N))
    z = np.concatenate((zl,z_nan,zr), axis=1)

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(131)
    #cax = fig.add_subplot(133)

    norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=0.5e-1)
    extent = [r_min,r_max,t_min,t_max]
    pos = ax.imshow(z, extent=extent, cmap='viridis', norm=norm, origin='lower', aspect='auto', interpolation='none', rasterized=True)

    # sub region of the original image
    x1, x2 = r_min, r_max
    if prob_id=='high_res':
        if args.early:
            y1, y2 = 0.2e5, 0.6e5
        else:
            y1, y2 = 2.5e5, 3.5e5
    elif prob_id=='b5':
        y1, y2 = 0.2e5, 0.6e5
    elif prob_id=='b200_hi':
        y1, y2 = 0.2e5, 0.6e5
    elif prob_id=='b5_hi':
        y1, y2 = 0.2e5, 0.6e5

    axins = fig.add_subplot(132)
    # inset axes....
    axins.imshow(z, extent=extent, cmap='viridis', norm=norm, origin='lower', aspect='auto', interpolation='none', rasterized=True)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # define corners to connect, clockwise from top left
    l1a, l2a = 2, 3 # start location of lines on original axis
    l1b, l2b = 1, 4 # end point of lines on zoom axis
    mark_inset(ax, axins, loc1a=l1a, loc1b=l1b, loc2a=l2a, loc2b=l2b, facecolor="none", edgecolor="black")

    ax.set_ylabel(r'time ($GM/c^3$)')
    ax.set_xlabel(r'$r$ ($r_g$)')
    axins.set_xlabel(r'$r$ ($r_g$)')

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.tick_params(axis='both', which='major')
    axins.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axins.tick_params(axis='both', which='major')

    cax = fig.add_axes([0.65, 0.12, 0.02, 0.75]) # [x0, y0, width, height]
    cbar = fig.colorbar(pos, cax=cax, extend='both', format=matplotlib.ticker.LogFormatterMathtext())
    cbar.ax.set_title(r'$\rho$')
    #plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=1200)
    plt.close()

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
# loc1, loc2 : {1, 2, 3, 4}
def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2



# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot density lump over time.')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs=1,
                        default=None,
                        help='data file to read, including path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--early',
                        action='store_true',
                        help='plot at early time (up to 1.2e5)')
    args = parser.parse_args()

    main(**vars(args))
