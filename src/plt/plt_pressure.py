#!/usr/bin/env python3
#
# plt_pressure.py
#
# Usage: python plt_pressure.py [options]
#
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import csv
import numpy as np
import matplotlib.pyplot as plt

# Athena++ modules
import athena_read

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)

    mesh_data = athena_read.athdf(data_dir + 'data/' + problem + ".cons.00000.athdf", quantities=['x2v'])
    theta = mesh_data['x2v']

    dens = np.load("dens_profile_th.npy")
    gpres = np.load("gpres_profile_th.npy")
    mpres = np.load("mpres_profile_th.npy")

    if problem=='high_res':
        mpres_scale = mpres*10.
        mpres_lab = r'$10P_{\rm mag}$'
    elif problem=='high_beta':
        mpres_scale = mpres*4.
        mpres_lab = r'$4P_{\rm mag}$'
    else:
        mpres_scale = mpres
        mpres_lab = r'$P_{\rm mag}$'

    lw = 1.5
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = [r'$\rho$', r'$P_{\rm gas}$', mpres_lab]
    _, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(theta, dens, linewidth=lw, color=colors[0], label=labels[0])
    ax1.plot([], [], linewidth=lw, color=colors[0], label=labels[0]) #ghost entry for ax1 legend
    ax1.plot(theta, gpres, linewidth=lw, color=colors[1], label=labels[1])
    ax1.plot(theta, mpres_scale, linewidth=lw, color=colors[2], label=labels[2])
    ax1.axvline(x=np.pi/2., color='k', linestyle=':')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$P$')
    ax2.set_ylabel(r'$\rho$')
    if args.grid:
        plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    leg = ax1.legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(data_dir + 'pressure.png')
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the scale height of the disk over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
