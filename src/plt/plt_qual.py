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
    av_Q_theta = []
    min_Q_theta = []
    max_Q_theta = []
    av_Q_phi = []
    min_Q_phi = []
    max_Q_phi = []
    theta_B = []
    with open('qual_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            t = float(row[0])
            tB = np.abs(float(row[1]))
            av_Qt = float(row[2])
            min_Qt = float(row[3])
            max_Qt = float(row[4])
            av_Qp = float(row[5])
            min_Qp = float(row[6])
            max_Qp = float(row[7])

            time.append(float(t)*5.)
            av_Q_theta.append(av_Qt)
            min_Q_theta.append(min_Qt)
            max_Q_theta.append(max_Qt)
            av_Q_phi.append(av_Qp)
            min_Q_phi.append(min_Qp)
            max_Q_phi.append(max_Qp)

            # Allow wrapping over 0,pi/2,pi etc to track evolution of theta_B more smoothly
            if len(theta_B)>0:
                prev_tB = theta_B[-1]
                lower_diff = tB - 90.
                upper_diff = tB + 90.
                if (np.abs(prev_tB-lower_diff)) < (np.abs(prev_tB-tB)):
                    theta_B.append(lower_diff)
                elif (np.abs(prev_tB-upper_diff)) < (np.abs(prev_tB-tB)):
                    theta_B.append(upper_diff)
                else:
                    theta_B.append(tB)
            else:
                theta_B.append(tB)

    y_label = r'in-plane $\theta_B$'
    y_var_name = 'theta_B'
    fig, ax = plt.subplots()
    ax.plot(time,theta_B)
    ax.set_xlabel('time since start (simulation units)')
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/' + y_var_name + '.png',dpi=1200)
    plt.close()

    if args.Qtheta:
        y_label = r'in-plane $Q_{\theta}$ at $r=25r_g$'
        y_var_name = 'Q_theta'
        fig, ax = plt.subplots()
        ax.plot(time, av_Q_theta)
        plt.fill_between(time,min_Q_theta,max_Q_theta,alpha=0.2)
        ax.set_xlabel('time since start (simulation units)')
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0)
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/' + y_var_name + '.png',dpi=1200)
        plt.close()

    if args.Qphi:
        y_label = r'in-plane $Q_{\phi}$ at $r=25r_g$'
        y_var_name = 'Q_phi'
        fig, ax = plt.subplots()
        ax.plot(time,av_Q_phi)
        plt.fill_between(time,min_Q_phi,max_Q_phi,alpha=0.2)
        ax.set_xlabel('time since start (simulation units)')
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0)
        plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig('/Users/paytonrodman/athena_sim/' + problem + '/' + y_var_name + '.png',dpi=1200)
        plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot various quality factors.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-Qt', '--Qtheta',
                        action="store_true",
                        help='plot theta Quality factor')
    parser.add_argument('-Qp', '--Qphi',
                        action="store_true",
                        help='plot phi Quality factor')
    args = parser.parse_args()

    main(**vars(args))
