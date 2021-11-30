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
            t_orb = float(row[1])
            b = float(row[2])
            time.append(t_orb*5.)
            beta.append(b)

    time, beta = zip(*sorted(zip(time, beta)))

    #beta_msk = [True if b<(100) else False for b in beta ]
    #beta = np.array(beta)
    #time = np.array(time)
    #beta = beta[beta_msk]
    #time = time[beta_msk]
    #beta.tolist()
    #time.tolist()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time,beta)
    ax.set_xlabel(r'time ($T_{5r_g}$)')
    ax.set_ylabel(r'average $\beta$')
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + 'beta' + '.png', dpi=1200)
    plt.close()

    if args.logy:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy(time,beta)
        ax.grid()
        ax.set_xlabel('time since start (simulation units)')
        ax.set_ylabel(r'average $\beta$')
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig(data_dir + 'logy_beta' + '.png', dpi=1200)
        plt.close()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot plasma beta over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot logy version')
    args = parser.parse_args()

    main(**vars(args))
