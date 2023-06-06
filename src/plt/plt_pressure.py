#!/usr/bin/env python3
#
# plt_pressure.py
#
# Usage: python plt_pressure.py [options]
#
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import numpy as np
import matplotlib.pyplot as plt
import re

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    n = len(args.prob_id) # number of data files to read
    if not args.prob_id:
        sys.exit('Must specify problem IDs')
    if not args.output:
        sys.exit('Must specify output file')

    gpres_o_mpres = [[] for _ in range(n)]
    x2v = [[] for _ in range(n)]
    labels = []
    colors = []
    for count,f in enumerate(args.prob_id):
        # get pre-defined labels and line colours for each simulation
        slash_list = [m.start() for m in re.finditer('/', f)]
        prob_id = f[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id)
        labels.append(l)
        colors.append(c)

        mesh_file = f + 'data/'+ prob_id + '.cons.00000.athdf'
        data_mesh = athena_read.athdf(mesh_file, quantities=['x2v'])
        x2v[count] = data_mesh['x2v']

        gpres_file = f + 'gpres_profile_th.npy'
        mpres_file = f + 'mpres_profile_th.npy'

        gpres = np.load(gpres_file, mmap_mode='r')
        mpres = np.load(mpres_file, mmap_mode='r')

        gpres_o_mpres[count] = mpres/gpres

    lw = 1.5
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(4,4))
    fig.subplots_adjust(hspace=0.0)
    ax.axhline(y=1, color="black", linestyle=":")
    for ii in range(n):
        ax.plot(x2v[ii], gpres_o_mpres[ii], linewidth=lw, color=colors[ii], label=labels[ii])


    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$P_{\rm mag}/P_{\rm gas}$')

    leg = ax.legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()

def directory_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the scale height of the disk over time.')
    parser.add_argument('-p', '--prob_id',
                        type=directory_path,
                        nargs='+',
                        default=None,
                        help='list of directory paths for each problem ID')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
