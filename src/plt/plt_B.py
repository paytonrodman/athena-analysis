#!/usr/bin/python
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import numpy as np
import scipy.stats
from ast import literal_eval
import argparse

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)
    time = []
    B_flux = []
    B_jet = []
    B_upper = []
    B_lower = []
    B_disk = []
    with open('B_strength_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            sim_t_i = float(row[0])
            orbit_t_i = float(row[1])
            b_flux_i = row[2]
            b_jet_i = row[3]
            b_up_i = row[4]
            b_low_i = row[5]
            b_disk_i = row[6]

            time.append(sim_t_i)
            B_flux.append(literal_eval(b_flux_i))
            B_jet.append(literal_eval(b_jet_i))
            B_upper.append(literal_eval(b_up_i))
            B_lower.append(literal_eval(b_low_i))
            B_disk.append(literal_eval(b_disk_i))

    time, B_flux, B_jet, B_upper, B_lower, B_disk = zip(*sorted(zip(time, B_flux, B_jet, B_upper, B_lower, B_disk)))

    B_flux = np.abs(np.array(B_flux))
    B_jet = np.abs(np.array(B_jet))
    B_upper = np.abs(np.array(B_upper))
    B_lower = np.abs(np.array(B_lower))
    B_disk = np.abs(np.array(B_disk))

    fig, axs = plt.subplots(2)
    axs[1].plot(time,B_flux[:,0],label=r'$B_{\Phi}$',linewidth=1)
    axs[1].plot(time,B_jet[:,0],label=r'$\overline{B}_{\rm jet}$',linewidth=1)
    axs[1].plot(time,B_upper[:,0],label=r'$\overline{B}_{\rm upper atmos}$',linewidth=1)
    axs[1].plot(time,B_lower[:,0],label=r'$\overline{B}_{\rm lower atmos}$',linewidth=1)
    axs[1].plot(time,B_disk[:,0],label=r'$\overline{B}_{\rm disk}$',linewidth=1)
    axs[1].set_xlabel(r'time ($GM/c^3$)')
    axs[1].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (lower)')
    axs[1].legend()

    axs[0].plot(time,B_flux[:,1],label=r'$B_{\Phi}$',linewidth=1)
    axs[0].plot(time,B_jet[:,1],label=r'$\overline{B}_{\rm jet}$',linewidth=1)
    axs[0].plot(time,B_upper[:,1],label=r'$\overline{B}_{\rm upper atmos}$',linewidth=1)
    axs[0].plot(time,B_lower[:,1],label=r'$\overline{B}_{\rm lower atmos}$',linewidth=1)
    axs[0].plot(time,B_disk[:,1],label=r'$\overline{B}_{\rm disk}$',linewidth=1)
    axs[0].set_xlabel(r'time ($GM/c^3$)')
    axs[0].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (upper)')
    axs[0].legend()

    for ax in fig.get_axes():
        ax.label_outer()
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        ax.minorticks_on()
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.tight_layout()
    plt.savefig(data_dir + 'B_strength' + '.png', dpi=1200)
    plt.close()


    # fig, axs = plt.subplots(2)
    # Bav = np.abs(np.array(Bav))
    # axs[1].plot(time,Bav[:,0],label=r'$B_{\Phi}$')
    # axs[1].plot(time,Bcc1_0sh,label=r'$B_{\rm midplane}$')
    # axs[1].plot(time,np.array(Bcc1_1sh)[:,0],label=r'$B_{1H/r}$')
    # axs[1].plot(time,np.array(Bcc1_2sh)[:,0],label=r'$B_{2H/r}$')
    # axs[1].plot(time,np.array(Bcc1_3sh)[:,0],label=r'$B_{3H/r}$')
    # axs[1].set_xlabel(r'time ($GM/c^3$)')
    # axs[1].set_ylabel(r'$\langle B\rangle_{r<100r_g}$ in disk (lower)')
    # axs[1].legend()
    #
    # axs[0].plot(time,Bav[:,1],label=r'$B_{\Phi}$')
    # axs[0].plot(time,Bcc1_0sh,label=r'$B_{\rm midplane}$')
    # axs[0].plot(time,np.array(Bcc1_1sh)[:,1],label=r'$B_{1H/r}$')
    # axs[0].plot(time,np.array(Bcc1_2sh)[:,1],label=r'$B_{2H/r}$')
    # axs[0].plot(time,np.array(Bcc1_3sh)[:,1],label=r'$B_{3H/r}$')
    # axs[0].set_xlabel(r'time ($GM/c^3$)')
    # axs[0].set_ylabel(r'$\langle B\rangle_{r<100r_g}$ in disk (upper)')
    # axs[0].legend()
    #
    # for ax in fig.get_axes():
    #     ax.label_outer()
    #     ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    #     ax.minorticks_on()
    #     ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    #     ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #
    # plt.tight_layout()
    # plt.savefig(data_dir + 'B_strength' + '.png', dpi=1200)
    # plt.close()
    #
    #
    #
    # if args.logy:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.semilogy(time,beta)
    #     ax.grid()
    #     ax.set_xlabel(r'time ($GM/c^3$)')
    #     ax.set_ylabel(r'average $\beta$ in disk')
    #     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    #     plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    #     plt.minorticks_on()
    #     plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #     plt.savefig(data_dir + 'logy_beta' + '.png', dpi=1200)
    #     plt.close()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot B value over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--logy',
                        action='store_true',
                        help='plot logy version')
    args = parser.parse_args()

    main(**vars(args))
