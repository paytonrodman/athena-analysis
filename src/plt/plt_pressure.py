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
    #ax.axvline(x=np.pi/2. - 3.*0.25, color="black", linestyle="--")
    #ax.axvline(x=np.pi/2. + 3.*0.25, color="black", linestyle="--")
    for ii in range(n):
        ax.plot(x2v[ii], gpres_o_mpres[ii], linewidth=lw, color=colors[ii], label=labels[ii])


    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$P_{\rm mag}/P_{\rm gas}$')

    leg = ax.legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()

    # directory containing data
    #problem  = args.prob_id
    #root_dir = '/Users/paytonrodman/athena-sim/'
    #data_dir = root_dir + problem + '/'
    #os.chdir(data_dir)
    #mesh_data = athena_read.athdf(data_dir + 'data/' + problem + ".cons.00000.athdf", quantities=['x2v'])
    #theta = mesh_data['x2v']

    #dens = np.load("dens_profile_th.npy")
    #gpres = np.load("gpres_profile_th.npy")
    #mpres = np.load("mpres_profile_th.npy")

    #lw = 1.5
    #colors = ['tab:orange', 'tab:green']
    #labels = [r'$P_{\rm gas}$', r'$P_{\rm mag}$']
    #_, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

    #ax1.plot(theta, gpres, linewidth=lw, color=colors[0], label=labels[0])
    #ax1.plot(theta, mpres, linewidth=lw, color=colors[1], label=labels[1])
    #ax1.axvline(x=np.pi/2., color='k', linestyle=':')
    #ax1.set_xlabel(r'$\theta$')
    #ax1.set_ylabel(r'$P$')
    #if args.grid:
    #    plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
    #    plt.minorticks_on()
    #    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    #leg = ax1.legend(loc='best')
    #for line in leg.get_lines():
    #    line.set_linewidth(4.0)

    #plt.savefig(data_dir + 'plots/pressure.png')
    #plt.close()
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the scale height of the disk over time.')
    parser.add_argument('-p', '--prob_id',
                        type=dir_path,
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
