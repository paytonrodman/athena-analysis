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
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Athena++ modules
import athena_read

def main(**kwargs):
    dens = []
    df = pd.read_csv(args.file, delimiter='\t', usecols=['sim_time', 'line_density'])
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
    t_min = 0
    t_max = time[-1]

    zl, zr = np.split(dens, 2, axis=1)
    z_nan = np.nan*np.ones((np.size(time),N))
    z = np.concatenate((zl,z_nan,zr), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=0.5e-1)
    pos = ax.imshow(z, extent=[r_min,r_max,t_min,t_max], cmap='viridis', norm=norm, origin='lower', aspect='auto', interpolation='none')

    ax.set_ylabel(r'time ($GM/c^3$)')
    ax.set_xlabel(r'$r$ ($r_g$)')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.tick_params(axis='both', which='major')

    cbar = fig.colorbar(pos, ax=ax, extend='both', format=matplotlib.ticker.LogFormatterMathtext())
    cbar.ax.set_title(r'$\rho$')
    plt.savefig(args.output, dpi=1200)
    plt.close()

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
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
