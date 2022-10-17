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
from ast import literal_eval
from scipy.interpolate import griddata
#import matplotlib.tri as tri
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Athena++ modules
import athena_read

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'

    os.chdir(data_dir)

    data_init = athena_read.athdf('data/' + args.prob_id + '.cons.00000.athdf', quantities=['x1v','x1f'])
    x1v = data_init['x1v']
    x1f = data_init['x1f']

    N = 30

    time = []
    time_orb = []
    dens = []
    with open('lump_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[1])
            d = row[3]
            time.append(t)
            dens.append(literal_eval(d))

    time, dens = zip(*sorted(zip(time, dens)))
    dens = np.asarray(dens)
    r_all = np.concatenate((-x1v,np.linspace(-5,5,N),x1v), axis=0)

    r_min = -290
    r_max = 290
    t_min = 0
    t_max = time[-1]

    #x = r_all
    #y = time
    #X, Y = np.meshgrid(x, y)
    #z = dens#[:,:896]
    zl, zr = np.split(dens, 2, axis=1)
    z_nan = np.nan*np.ones((np.size(time),N))
    z = np.concatenate((zl,z_nan,zr), axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=0.5e-1)
    #pos = ax.contourf(X, Y, np.log(z), levels=20, cmap='viridis')
    pos = ax.imshow(z, extent=[r_min,r_max,t_min,t_max], cmap='viridis', norm=norm, origin='lower', aspect='auto', interpolation='none')
    #pos = ax.pcolormesh(X, Y, dens[:-1,:-1], cmap='viridis', norm=norm)
    #ax.axis('auto')

    ax.set_ylabel(r'time ($GM/c^3$)')
    ax.set_xlabel(r'$r$ ($r_g$)')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.tick_params(axis='both', which='major')

    cbar = fig.colorbar(pos, ax=ax, extend='both', format=matplotlib.ticker.LogFormatterMathtext())
    #cbar.ax.set_ylabel(r'$\rho$', rotation=0)
    cbar.ax.set_title(r'$\rho$')
    plt.savefig(data_dir + 'plots/lump' + '.png', dpi=1200)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot density lump over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
