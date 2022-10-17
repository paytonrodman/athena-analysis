#!/usr/bin/python
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import matplotlib.pyplot as plt
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
        args.sharex=True

    labels = []
    colors = []
    for f in args.file:
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id)
        labels.append(l)
        colors.append(c)

    t_lists = [[] for _ in range(n)]
    qt_lists = [[] for _ in range(n)]
    qp_lists = [[] for _ in range(n)]
    tb_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        df = pd.read_csv(f, delimiter='\t', usecols=['sim_time', 'Q_theta', 'Q_phi', 'theta_B'])
        t = df['sim_time'].to_list()
        Qt = df['Q_theta'].to_list()
        Qp = df['Q_phi'].to_list()
        tB = df['theta_B'].to_list()
        t_lists[count] = t
        qt_lists[count] = Qt
        qp_lists[count] = Qp
        tb_lists[count] = tB

    for ii in range(n):
        t_lists[ii], qt_lists[ii], qp_lists[ii], tb_lists[ii] = zip(*sorted(zip(t_lists[ii], qt_lists[ii], qp_lists[ii], tb_lists[ii])))

    lw = 1.5
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4,6))
    fig.subplots_adjust(hspace=0.0)  # adjust space between axes
    if args.sharex:
        axs[0].axhline(y=6, color='k',linestyle='dotted')
        axs[0].axhline(y=10, color='k',linestyle='dashed')
        axs[1].axhline(y=20, color='k',linestyle='dashed')

        for ii in range(n):
            axs[0].plot(t_lists[ii], qt_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            axs[1].plot(t_lists[ii], qp_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            axs[2].plot(t_lists[ii], tb_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])

        axs[0].set_ylabel(r'$\langle Q_{\theta}\rangle$')
        axs[1].set_ylabel(r'$\langle Q_{\phi}\rangle$')
        axs[2].set_ylabel(r'$\langle \theta_{B}\rangle$')
        for ax in axs:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        axs[-1].set_xlabel(r'time ($GM/c^3$)')
        leg = axs[0].legend(loc='best')
        for line in leg.get_lines():
            line.set_linewidth(4.0)

        plt.savefig(args.output, bbox_inches='tight')
        plt.close()
    else:
        ax0_a = axs[0].twiny()  # instantiate a second axes that shares the same x-axis
        ax1_a = axs[1].twiny()
        ax2_a = axs[2].twiny()

        axs[0].axhline(y=6, color='k',linestyle='dotted')
        axs[0].axhline(y=10, color='k',linestyle='dashed')
        axs[1].axhline(y=20, color='k',linestyle='dashed')

        axs[0].plot(t_lists[0], qt_lists[0], color=colors[0], label=labels[0], linewidth=lw)
        axs[0].plot([], [], color=colors[1], label=labels[1]) # ghost plot for color2 label entry
        ax0_a.plot(t_lists[1], qt_lists[1], color=colors[1], label=labels[1], linewidth=lw)

        axs[1].plot(t_lists[0], qp_lists[0], color=colors[0], label=labels[0], linewidth=lw)
        axs[1].plot([], [], color=colors[1], label=labels[1]) # ghost plot for color2 label entry
        ax1_a.plot(t_lists[1], qp_lists[1], color=colors[1], label=labels[1], linewidth=lw)

        axs[2].plot(t_lists[0], tb_lists[0], color=colors[0], label=labels[0], linewidth=lw)
        axs[2].plot([], [], color=colors[1], label=labels[1]) # ghost plot for color2 label entry
        ax2_a.plot(t_lists[1], tb_lists[1], color=colors[1], label=labels[1], linewidth=lw)

        ax0_a.set_xlabel(r'time ($GM/c^3$)', color=colors[1])
        ax0_a.tick_params(axis='x', labelcolor=colors[1])
        axs[2].set_xlabel(r'time ($GM/c^3$)', color=colors[0])
        axs[2].tick_params(axis='x', labelcolor=colors[0])

        axs[0].set_ylabel(r'$\langle Q_{\theta}\rangle$')
        axs[1].set_ylabel(r'$\langle Q_{\phi}\rangle$')
        axs[2].set_ylabel(r'$\langle \theta_{B}\rangle$')

        for ax in [axs[0],axs[1],axs[2],ax0_a,ax1_a,ax2_a]:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
            ax.set_xlim(left=0)
        for ax in [axs[0],ax0_a,axs[2],ax2_a]:
            ax.set_ylim(bottom=0)
        for ax in [ax1_a,ax2_a]:
            ax.set_xticks([])

        leg = axs[0].legend(loc='best')
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
    parser.add_argument('--sharex',
                        action='store_true',
                        help='share x (time) axis')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
