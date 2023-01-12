#!/usr/bin/python
# Python standard modules
import argparse
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

    labels = []
    colors = []
    for f in args.file:
        slash_list = [m.start() for m in re.finditer('/', f.name)]
        prob_id = f.name[slash_list[-2]+1:slash_list[-1]]
        l,c,_ = AAT.problem_dictionary(prob_id, args.pres)
        labels.append(l)
        colors.append(c)

    t_lists = [[] for _ in range(n)]
    s_lists = [[] for _ in range(n)]
    for count,f in enumerate(args.file):
        df = pd.read_csv(f, delimiter='\t', usecols=['sim_time', 'scale_height'])
        t = df['sim_time'].to_list()
        s = df['scale_height'].to_list()
        #t_lists[count] = t
        t_lists[count] = [ti/1e5 for ti in t] # convert time to units of 10^5 GM/c3
        s_lists[count] = s

    for ii in range(n):
        t_lists[ii], s_lists[ii] = zip(*sorted(zip(t_lists[ii], s_lists[ii])))

    lw = 1.5
    x_label = r'time [$10^5~GM/c^3$]'
    y_label = r'$\langle H \rangle$'

    fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True, sharex=True)
    for ii in range(n):
        if args.logx:
            ax1.semilogx(t_lists[ii], s_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
        elif args.logy:
            ax1.semilogy(t_lists[ii], s_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
        else:
            ax1.plot(t_lists[ii], s_lists[ii], linewidth=lw, color=colors[ii], label=labels[ii])
            ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
            ax1.set_ylim(bottom=0, top=0.5)
            ax1.set_xlim(left=0)
    ax1.set_xlabel(x_label)#, x=0.5, y=-0.03)
    ax1.set_ylabel(y_label)

    if args.grid:
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
        ax1.minorticks_on()
        ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    leg = ax1.legend(loc='best')
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.savefig(args.output, bbox_inches='tight')
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot plasma beta over time.')
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
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    parser.add_argument('--pres',
                        action='store_true',
                        help='make presentation-quality image')
    args = parser.parse_args()

    main(**vars(args))
