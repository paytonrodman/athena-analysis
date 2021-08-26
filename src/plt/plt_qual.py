#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    data_dir = "/Users/paytonrodman/athena_sim/" + problem + "/"
    os.chdir(data_dir)

    filename_csv = args.input_file

    time = []
    theta_B = []
    Qt_lc,Qt_av,Qt_uc = [],[],[]
    Qp_lc,Qp_av,Qp_uc = [],[],[]
    with open(filename_csv, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t_i = float(row[0])
            tB_i = float(row[1])
            Qt_lc_i = float(row[2])
            Qt_av_i = float(row[3])
            Qt_uc_i = float(row[4])
            Qp_lc_i = float(row[5])
            Qp_av_i = float(row[6])
            Qp_uc_i = float(row[7])

            time.append(float(t_i)*5.)
            theta_B.append(tB_i)
            Qt_lc.append(Qt_lc_i)
            Qt_av.append(Qt_av_i)
            Qt_uc.append(Qt_uc_i)
            Qp_lc.append(Qp_lc_i)
            Qp_av.append(Qp_av_i)
            Qp_uc.append(Qp_uc_i)


            # Allow wrapping over 0,pi/2,pi etc to track evolution of theta_B more smoothly
            #if len(tB)>1:
            #    prev_tB = tB[-1]
            #    lower_diff = tB_i - 90.
            #    upper_diff = tB_i + 90.
            #    if (np.abs(prev_tB-lower_diff)) < (np.abs(prev_tB-tB_i)):
            #        theta_B.append(lower_diff)
            #    elif (np.abs(prev_tB-upper_diff)) < (np.abs(prev_tB-tB_i)):
            #        theta_B.append(upper_diff)
            #    else:
            #        theta_B.append(tB_i)
            #else:
            #    theta_B.append(tB_i)

    make_plot(time,theta_B,r'$\theta_B$','theta_B',data_dir)
    make_plot_CI(time,Qt_av,Qt_lc,Qt_uc,r'average $Q_{\theta}$','Q_theta',data_dir)
    make_plot(time,Qp_av,Qp_lc,Qp_uc,r'average $Q_{\phi}$','Q_phi',data_dir)

def make_plot(x,y,ylabel,yname,data_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    ax.set_xlabel('time since start (simulation units)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + yname + '.png',dpi=1200)
    plt.close()

def make_plot_CI(x,y,y_CI_low,y_CI_high,ylabel,yname,data_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    plt.fill_between(x,y_CI_low,y_CI_high,alpha=0.2)
    ax.set_xlabel('time since start (simulation units)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(data_dir + yname + '.png',dpi=1200)
    plt.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot various quality factors.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('input_file',
                        default='qual_with_time_0_148_127_128.csv',
                        help='name of the file with quality data to be analysed, including extension, e.g. qual_with_time.csv (default=qual_with_time_0_148_127_128.csv)')
    parser.add_argument('-s', '--slice',
                        action="store_true",
                        help='plot a vertical slice (phi=0) of phi and/or theta Quality factors')

    args = parser.parse_args()

    main(**vars(args))
