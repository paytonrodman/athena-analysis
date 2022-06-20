#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import numpy as np
import scipy.stats
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)
    time = []
    beta = []
    with open('beta_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            b = float(row[2])
            time.append(t)
            beta.append(b)

    time, beta = zip(*sorted(zip(time, beta)))

    lw = 1.5

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    if args.logy:
        ax.semilogy(time, beta, linewidth=lw)
    else:
        ax.plot(time, beta, linewidth=lw)
    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(r'$\langle\beta\rangle$')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if args.grid:
        plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    plt.savefig(data_dir + 'beta' + '.png', dpi=1200)
    plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot plasma beta over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot logy version')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
