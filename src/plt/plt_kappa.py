#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import numpy as np
import scipy.stats
import pandas as pd
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)
    time = []
    k_jet = []
    k_upatmos = []
    k_lowatmos = []
    k_disk = []
    with open('curvature_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            jet = np.fromstring(row[2].strip("[]"), sep=',')
            upatmos = np.fromstring(row[3].strip("[]"), sep=',')
            lowatmos = np.fromstring(row[4].strip("[]"), sep=',')
            disk = np.fromstring(row[5].strip("[]"), sep=',')

            time.append(t)
            k_jet.append(jet)
            k_upatmos.append(upatmos)
            k_lowatmos.append(lowatmos)
            k_disk.append(disk)

    time, k_jet, k_upatmos, k_lowatmos, k_disk = zip(*sorted(zip(time, k_jet, k_upatmos, k_lowatmos, k_disk)))

    time = np.asarray(time)
    k_jet = np.asarray(k_jet)
    k_upatmos = np.asarray(k_upatmos)
    k_lowatmos = np.asarray(k_lowatmos)
    k_disk = np.asarray(k_disk)

    if args.averaged:
        window = 100
        av_k_jet = pd.Series(k_jet[:,0]).rolling(window).mean()
        av_k_upatmos = pd.Series(k_upatmos[:,0]).rolling(window).mean()
        av_k_lowatmos = pd.Series(k_lowatmos[:,0]).rolling(window).mean()
        av_k_disk = pd.Series(k_disk[:,0]).rolling(window).mean()

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(12,4))
    colors = ['#00b6ff', '#217aa9', '#1e4258', '#05070a']
    labels = ['jet', 'upper atmos.', 'lower atmos.', 'disk']
    lw = 1.5
    a = 0.7
    if args.logy:
        if args.averaged:
            ax.semilogy(time, av_k_jet, linewidth=lw, color=colors[0], label=labels[0])
            ax.semilogy(time, av_k_upatmos, linewidth=lw, color=colors[1], label=labels[1])
            ax.semilogy(time, av_k_lowatmos, linewidth=lw, color=colors[2], label=labels[2])
            ax.semilogy(time, av_k_disk, linewidth=lw, color=colors[3], label=labels[3])
        else:
            ax.semilogy(time, k_disk[:,0], linewidth=lw, color=colors[0], label=labels[0], alpha=a)
            ax.semilogy(time, k_lowatmos[:,0], linewidth=lw, color=colors[1], label=labels[1], alpha=a)
            ax.semilogy(time, k_upatmos[:,0], linewidth=lw, color=colors[2], label=labels[2], alpha=a)
            ax.semilogy(time, k_jet[:,0], linewidth=lw, color=colors[3], label=labels[3], alpha=a)
    else:
        if args.averaged:
            ax.plot(time, av_k_jet, linewidth=lw, color=colors[0], label=labels[0])
            ax.plot(time, av_k_upatmos, linewidth=lw, color=colors[1], label=labels[1])
            ax.plot(time, av_k_lowatmos, linewidth=lw, color=colors[2], label=labels[2])
            ax.plot(time, av_k_disk, linewidth=lw, color=colors[3], label=labels[3])
        else:
            ax.plot(time, k_disk[:,0], linewidth=lw, color=colors[0], label=labels[0], alpha=a)
            ax.plot(time, k_lowatmos[:,0], linewidth=lw, color=colors[1], label=labels[1], alpha=a)
            ax.plot(time, k_upatmos[:,0], linewidth=lw, color=colors[2], label=labels[2], alpha=a)
            ax.plot(time, k_jet[:,0], linewidth=lw, color=colors[3], label=labels[3], alpha=a)

    ax.set_xlabel(r'time ($GM/c^3$)')
    ax.set_ylabel(r'$\langle\kappa\rangle$')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if args.grid:
        plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    leg = ax.legend(loc='lower right')
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.savefig(data_dir + 'kappa.png')
    plt.close()

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot plasma beta over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot logy version')
    parser.add_argument('--averaged',
                        action='store_true',
                        help='plot averaged (smoothed) data')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    args = parser.parse_args()

    main(**vars(args))
