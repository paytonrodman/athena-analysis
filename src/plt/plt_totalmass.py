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
        # get pre-defined labels and line colours for each simulation
        slash_list = [m.start() for m in re.finditer('/', f)]
        prob_id = f[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

    t_lists = [[] for _ in range(n)]
    mass_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        df = pd.read_csv(f, skiprows=1, sep='\s+', escapechar='#')

        mass_tot = df['[3]=mass'].to_list()
        t = df[' [1]=time'].to_list()

        if args.orbits:
            t_lists[count] = [AAT.calculate_orbit_time(ti) for ti in t]
        else:
            t_lists[count] = [ti/1e5 for ti in t] # convert time to units of 10^5 GM/c3
        mass_lists[count] = mass_tot

    for ii in range(n):
        t_lists[ii], mass_lists[ii] = zip(*sorted(zip(t_lists[ii], mass_lists[ii])))

    lw = 1.5
    if args.orbits:
        x_label = r'Time [ISCO Orbits]'
    else:
        x_label = r'Time [$10^5~GM/c^3$]'
    y_label = r'Total Mass'

    _, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True, sharex=True)
    for ii in range(n):
        if args.logy:
            ax1.semilogy(t_lists[ii], mass_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
        else:
            ax1.plot(t_lists[ii], mass_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
    ax1.set_xlabel(x_label)#, x=0.5, y=-0.03)
    ax1.set_ylabel(y_label)

    leg = ax1.legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.savefig(args.output, bbox_inches='tight')
    plt.close()


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
