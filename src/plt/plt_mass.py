#!/usr/bin/python
# Python standard modules
import argparse
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
    n = len(args.file) # number of data files to read
    if not args.file:
        sys.exit('Must specify data files')
    if not args.output:
        sys.exit('Must specify output file')
    if n>2 and args.sharex is False:
        raise ValueError('Cannot have separate time axes for n>2. Use --sharex')
    if n==1 and args.sharex is False:
        args.sharex = True

    labels = []
    colors = []
    for f in args.file:
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

    t_lists = [[] for _ in range(n)]
    m_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        df = pd.read_csv(f, delimiter='\t', usecols=['file_time', 'sim_time', 'mass_flux'])
        t = df['sim_time'].to_list()
        m = df['mass_flux'].to_list()
        #t_lists[count] = t
        t_lists[count] = [ti/1e5 for ti in t]
        m_lists[count] = m

    for ii in range(n):
        t_lists[ii], m_lists[ii] = zip(*sorted(zip(t_lists[ii], m_lists[ii])))


    lw = 1.5
    x_label = r'time [$10^5~GM/c^3$]'
    y_label = r'$\dot{M}$'

    if args.sharex:
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, sharex=True)
        for ii in range(n):
            if args.logx:
                ax.semilogx(t_lists[ii], m_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            elif args.logy:
                ax.semilogy(t_lists[ii], m_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            else:
                ax.plot(t_lists[ii], m_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
                ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useMathText=True)
        plt.axhline(color="grey")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0,top=2)

        leg = ax.legend(loc='best')

        if args.grid:
            ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    else:
        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
        axs = [ax1,ax2]
        ax1.plot(t_lists[0], m_lists[0], color=colors[0], label=labels[0], linewidth=lw, rasterized=True)
        ax1.plot([], [], color=colors[1], label=labels[1]) # ghost plot for color2 label entry
        ax2.plot(t_lists[1], m_lists[1], color=colors[1], label=labels[1], linewidth=lw, rasterized=True)
        plt.axhline(color="grey")
        ax1.set_xlabel(x_label, color=colors[0])
        ax2.set_xlabel(x_label, color=colors[1])
        ax1.tick_params(axis='x', labelcolor=colors[0])
        ax2.tick_params(axis='x', labelcolor=colors[1])
        ax1.set_ylabel(y_label)
        for ax in axs:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=-0.5,top=2)
            if args.grid:
                ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
                ax.minorticks_on()
                ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
        leg = ax1.legend(loc='best')

    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot magnetic field curvature, kappa, over time for different disk regions.')
    parser.add_argument('-f', '--file',
                        type=argparse.FileType('r'),
                        nargs='+',
                        default=None,
                        help='list of data files to read, including path')
    parser.add_argument('-o', '--output',
                        type=str,
                        default=None,
                        help='name of plot to be created, including path')
    parser.add_argument('--logx',
                        action='store_true',
                        help='plot logx version')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot logy version')
    parser.add_argument('--sharex',
                        action='store_true',
                        help='share x (time) axis')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
