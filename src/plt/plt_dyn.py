#!/usr/bin/env python3
#
# plt_dyn.py
#
# A program to plot the value of dynamo coefficients, from data generated by calc_dyn.py
#
# Usage: python plt_dyn.py [options]
#
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')

# Other Python modules
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd

def main(**kwargs):
    alpha = []
    C = []

    df = pd.read_csv(args.file, delimiter='\t', usecols=['sim_time', 'orbit_time', 'alpha', 'C'])
    t = df['sim_time'].to_list()
    t_orb = df['orbit_time'].to_list()
    a = df['alpha'].to_list()
    c = df['C'].to_list()
    for ai in a: # alpha given as [a_r,a_theta,a_phi]
        a_list = np.fromstring(ai.strip("[]"), sep=', ')
        alpha.append(a_list)
    for ci in c: # C given as [c_r,c_theta,c_phi]
        c_list = np.fromstring(ci.strip("[]"), sep=', ')
        C.append(c_list)
    time = t
    time_orb = t_orb

    time, time_orb, alpha, C = zip(*sorted(zip(time, time_orb, alpha, C)))

    time = np.asarray(time)
    time_orb = np.asarray(time_orb)
    alpha = np.asarray(alpha)
    C = np.asarray(C)

    if args.path: # plot path for last 50 orbits
        time_mask = (time_orb > (np.max(time_orb) - 100.)) & (time_orb < np.max(time_orb))
    else:
        time_mask = (time_orb > 50.) # plot all data later than 50 orbits

    alpha = alpha[time_mask]
    C = C[time_mask]
    time_orb = time_orb[time_mask]
    math = [r'$r$', r'$\theta$', r'$\phi$']

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4), constrained_layout=True)
    xlab = r'Offset ($C$)'
    ylab = r'$\alpha_d$'
    for num in range(0,3):
        x = alpha[:,num]
        y = C[:,num]
        m = math[num]
        t = time_orb

        axs[num].axhline(y=0, color='grey', linestyle='-', linewidth=1)
        axs[num].axvline(x=0, color='grey', linestyle='-', linewidth=1)

        #axs[num].plot(x,y,'k.',markersize=5,alpha=0.3)
        axs[num].scatter(x,y,c=t,cmap='viridis',alpha=0.3)
        if args.path:
            axs[num].plot(x,y,'r')

        x_max = np.abs(axs[num].get_xlim()).max()
        y_max = np.abs(axs[num].get_ylim()).max()
        axs[num].set_xlim(-x_max, x_max)
        axs[num].set_ylim(-y_max, y_max)

        at = AnchoredText(m, prop=dict(size=15), frameon=True, loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axs[num].add_artist(at)

    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    axs[0].set_ylabel(ylab)
    axs[1].set_xlabel(xlab)
    plt.savefig(args.output,dpi=args.dpi)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs=1,
                        default=None,
                        help='data file to read, including path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('-dpi', '--dpi',
                        type=int,
                        default=250,
                        help='dots per inch')
    parser.add_argument('-p', '--path',
                        action="store_true",
                        help='specify whether to plot the path over time')
    args = parser.parse_args()

    main(**vars(args))
