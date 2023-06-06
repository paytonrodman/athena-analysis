#!/usr/bin/python
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import matplotlib.pyplot as plt
import pandas as pd
import re

# Athena++ modules
import AAT

def main(**kwargs):
    n = len(args.file) # number of data files to read
    if not args.file:
        sys.exit('Must specify data (hst) files')
    if not args.output:
        sys.exit('Must specify output file')

    labels = []
    colors = []
    for f in args.file:
        slash_list = [m.start() for m in re.finditer('/', f)]
        prob_id = f[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

    t_lists = [[] for _ in range(n)]
    ME_lists = [[] for _ in range(n)]
    ME_1_lists = [[] for _ in range(n)]
    ME_2_lists = [[] for _ in range(n)]
    ME_3_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        df = pd.read_csv(f, skiprows=1, sep='\s+', escapechar='#')

        # history file contains some duplicates due to restarts, remove them here
        df.drop_duplicates(subset=[' [1]=time'], keep='last', inplace=True)

        ME_1 = df['[11]=1-ME'].to_list()
        ME_2 = df['[12]=2-ME'].to_list()
        ME_3 = df['[13]=3-ME'].to_list()
        ME_tot = [ME_1[i] + ME_2[i] + ME_3[i] for i in range(len(ME_1))]

        t = df[' [1]=time'].to_list()

        if args.orbits:
            t_lists[count] = [AAT.calculate_orbit_time(ti) for ti in t]
        else:
            t_lists[count] = [ti/1e5 for ti in t] # convert time to units of 10^5 GM/c3
        ME_1_lists[count]  = ME_1
        ME_2_lists[count]  = ME_2
        ME_3_lists[count]  = ME_3
        ME_lists[count] = ME_tot

    for ii in range(n):
        t_lists[ii], ME_1_lists[ii], ME_2_lists[ii], ME_3_lists[ii], ME_lists[ii] = zip(*sorted(zip(t_lists[ii], ME_1_lists[ii], ME_2_lists[ii], ME_3_lists[ii], ME_lists[ii])))

    lw = 1.5
    if args.orbits:
        x_label = r'Time [ISCO Orbits]'
    else:
        x_label = r'Time [$10^5~GM/c^3$]'
    y_label = r'Magnetic Energy'

    _, axs = plt.subplots(nrows=1, ncols=n, constrained_layout=True, sharex=True)
    for ii in range(n):
        if n>1:
            ax = axs[ii]
        else:
            ax = axs
        if args.logy:
            ax.semilogy(t_lists[ii], ME_1_lists[ii], linewidth=1, linestyle='dotted', color=colors[ii], label=r'$r$')
            ax.semilogy(t_lists[ii], ME_2_lists[ii], linewidth=1, linestyle='dashed', color=colors[ii], label=r'$\theta$')
            ax.semilogy(t_lists[ii], ME_3_lists[ii], linewidth=1, linestyle='dashdot', color=colors[ii], label=r'$\phi$')
            ax.semilogy(t_lists[ii], ME_lists[ii], linewidth=lw, color=colors[ii], label='Total')
            ax.set_ylim(bottom=0.01)
        else:
            ax.plot(t_lists[ii], ME_1_lists[ii], linewidth=1, linestyle='dotted', color=colors[ii], label=r'$r$')
            ax.plot(t_lists[ii], ME_2_lists[ii], linewidth=1, linestyle='dashed', color=colors[ii], label=r'$\theta$')
            ax.plot(t_lists[ii], ME_3_lists[ii], linewidth=1, linestyle='dashdot', color=colors[ii], label=r'$\phi$')
            ax.plot(t_lists[ii], ME_lists[ii], linewidth=lw, color=colors[ii], label='Total')

        ax.set_xlabel(x_label)#, x=0.5, y=-0.03)
        ax.set_ylabel(y_label)
        ax.set_title(labels[ii])

        leg = ax.legend(loc='best')
        #for line in leg.get_lines():
        #    line.set_linewidth(2.0)
    plt.savefig(args.output, bbox_inches='tight')
    plt.close()

def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot total magnetic energy over time.')
    parser.add_argument('-f', '--file',
                        type=str,
                        nargs='+',
                        default=None,
                        help='list of data files to read, including path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot logy version')
    parser.add_argument('--orbits',
                        action='store_true',
                        help='plot against number of ISCO orbits')
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
