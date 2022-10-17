#!/usr/bin/python
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import csv
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.stats

# Athena++ modules
import athena_read

def main(**kwargs):
    n = len(args.prob_id) # number of data files to read
    print(args.prob_id)
    if not args.prob_id:
        sys.exit('Must specify problem IDs')
    if not args.output:
        sys.exit('Must specify output file')
    if n>2 and args.sharex is False:
        raise ValueError('Cannot have separate time axes for n>2. Use --sharex')
    if n==1 and args.sharex is False:
        args.sharex = True

    dens = [[] for _ in range(n)]
    mom1 = [[] for _ in range(n)]
    temp = [[] for _ in range(n)]
    x1v = [[] for _ in range(n)]
    labels = []
    colors = []
    for count,f in enumerate(args.prob_id):
        slash_list = [m.start() for m in re.finditer('/', f)]
        prob_id = f[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id)
        labels.append(l)
        colors.append(c)

        mesh_file = f + 'data/'+ prob_id + '.cons.00000.athdf'
        data_mesh = athena_read.athdf(mesh_file, quantities=['x1v'])
        x1v[count] = data_mesh['x1v']

        dens_file = f + 'dens_profile.npy'
        mom1_file = f + 'mom1_profile.npy'
        temp_file = f + 'temp_profile.npy'

        dens[count] = np.load(dens_file, mmap_mode='r')
        mom1[count] = np.load(mom1_file, mmap_mode='r')
        temp[count] = np.load(temp_file, mmap_mode='r')

    ylabels = [r'$\langle \rho \rangle$', r'$\langle v_r \rangle$', r'$\langle \frac{P}{\rho c^2} \rangle$']
    lw = 1.5

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4,6))
    fig.subplots_adjust(hspace=0.0)
    for ii in range(n):
        if args.logr:
            axs[0].semilogx(x1v[ii], dens[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            axs[1].semilogx(x1v[ii], mom1[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            axs[2].semilogx(x1v[ii], temp[ii], linewidth=lw, color=colors[ii], label=labels[ii])
        else:
            axs[0].plot(x1v[ii], dens[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            axs[1].plot(x1v[ii], mom1[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            axs[2].plot(x1v[ii], temp[ii], linewidth=lw, color=colors[ii], label=labels[ii])
    for ii in range(3):
        axs[ii].set_ylabel(ylabels[ii])
        if not args.logr:
            axs[ii].set_xlim(left=0)
            axs[ii].ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)

    axs[-1].set_xlabel(r'radius ($r_g$)')

    leg = axs[0].legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot magnetic field curvature, kappa, over time for different disk regions.')
    parser.add_argument('-p', '--prob_id',
                        type=dir_path,
                        nargs='+',
                        default=None,
                        help='list of directory paths for each problem ID')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--logr',
                        action='store_true',
                        help='plot r in log')
    args = parser.parse_args()

    main(**vars(args))
