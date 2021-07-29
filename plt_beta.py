#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import athena_read
import glob
import re
import csv
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    data_dir = "/Users/paytonrodman/athena_sim/" + problem + "/"
    os.chdir(data_dir)
    time = []
    beta = []
    with open('beta_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = row[0]
            b = row[1]
            time.append(float(t)*5.)
            beta.append(float(b))

    fig, ax = plt.subplots()
    ax.plot(time,beta)
    ax.set_xlabel('time since start (simulation units)')
    ax.set_ylabel(r'average in-plane $\beta$ for $r<45r_g$')
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/beta.png',dpi=1200)
    plt.close()

    if args.logy:
        fig, ax = plt.subplots()
        ax.semilogy(time,beta)
        ax.grid()
        ax.set_xlabel('time since start (simulation units)')
        ax.set_ylabel(r'average in-plane $\beta$ for $r<45r_g$')
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/logy_beta.png',dpi=1200)
        plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot plasma beta over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--logy',
                        action="store_true",
                        help='plot logy version')
    args = parser.parse_args()

    main(**vars(args))
