#!/usr/bin/python
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Athena++ modules
import AAT

def main(**kwargs):
    n = len(args.prob_id) # number of data files to read
    if not args.prob_id:
        raise ValueError('Must specify problem directories with -p')
    if not args.output:
        raise ValueError('Must specify output file with -o')

    t1 = [[] for _ in range(n)]
    t2 = [[] for _ in range(n)]
    t3 = [[] for _ in range(n)]
    mass = [[] for _ in range(n)]
    beta = [[] for _ in range(n)]
    scale = [[] for _ in range(n)]
    labels = []
    colors = []
    for count,f in enumerate(args.prob_id):
        slash_list = [m.start() for m in re.finditer('/', f)]
        prob_id = f[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

        mass_file = f + 'mass_with_time.csv'
        beta_file = f + 'beta_with_time.csv'
        scale_file = f + 'scale_with_time.csv'

        df = pd.read_csv(mass_file, delimiter='\t', usecols=['sim_time', 'mass_flux'])
        t = df['sim_time'].to_list()
        m = df['mass_flux'].to_list()
        for mi in m:
            if type(mi) is str:
                mi = np.fromstring(mi.strip("[]"), sep=', ')
            mass[count].append(mi)
        t1[count] = t
        #mass[count] = m

        df = pd.read_csv(beta_file, delimiter='\t', usecols=['sim_time', 'plasma_beta'])
        t = df['sim_time'].to_list()
        b = df['plasma_beta'].to_list()
        t2[count] = t
        beta[count] = b

        df = pd.read_csv(scale_file, delimiter='\t', usecols=['sim_time', 'scale_height'])
        t = df['sim_time'].to_list()
        s = df['scale_height'].to_list()
        t3[count] = t
        scale[count] = s

    for ii in range(n):
        t1[ii],mass[ii] = zip(*sorted(zip(t1[ii],mass[ii])))
        t2[ii],beta[ii] = zip(*sorted(zip(t2[ii],beta[ii])))
        t3[ii],scale[ii] = zip(*sorted(zip(t3[ii],scale[ii])))

    ylabels = [r'$\langle \dot{M} \rangle$', r'$\langle \beta \rangle$', r'$\langle H \rangle$']
    lw = 1.5

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4,6))
    fig.subplots_adjust(hspace=0.0)
    for ii in range(n):
        axs[0].plot(t1[ii], mass[ii], linewidth=lw, color=colors[ii], label=labels[ii])
        axs[1].semilogy(t2[ii], beta[ii], linewidth=lw, color=colors[ii], label=labels[ii])
        axs[2].plot(t3[ii], scale[ii], linewidth=lw, color=colors[ii], label=labels[ii])

    for ii in range(3):
        axs[ii].set_ylabel(ylabels[ii])
    for ax in axs:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
        ax.set_xlim(left=0)

    axs[0].set_ylim(bottom=0, top=2)
    axs[1].set_ylim(bottom=1e0)
    axs[2].set_ylim(bottom=0.2, top=0.37)

    axs[-1].set_xlabel(r'time ($GM/c^3$)')

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
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
