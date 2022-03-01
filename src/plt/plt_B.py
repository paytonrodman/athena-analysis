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
    if args.component not in ['r','theta','phi','all']:
        sys.exit('Must specify a valid B component')

    # directory containing data
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + args.prob_id + '/'
    os.chdir(data_dir)
    time = []
    abs_B_flux = []
    abs_B_jet = []
    abs_B_upper = []
    abs_B_lower = []
    abs_B_disk = []
    sign_B_flux = []
    sign_B_jet = []
    sign_B_upper = []
    sign_B_lower = []
    sign_B_disk = []
    with open('B_strength_with_time_'+args.component+'.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            sim_t_i = float(row[0])
            #orbit_t_i = float(row[1])
            abs_b_flux_i = row[2]
            abs_b_jet_i = row[3]
            abs_b_up_i = row[4]
            abs_b_low_i = row[5]
            abs_b_disk_i = row[6]
            sign_b_flux_i = row[7]
            sign_b_jet_i = row[8]
            sign_b_up_i = row[9]
            sign_b_low_i = row[10]
            sign_b_disk_i = row[11]

            time.append(sim_t_i)
            abs_B_flux.append(literal_eval(abs_b_flux_i))
            abs_B_jet.append(literal_eval(abs_b_jet_i))
            abs_B_upper.append(literal_eval(abs_b_up_i))
            abs_B_lower.append(literal_eval(abs_b_low_i))
            abs_B_disk.append(literal_eval(abs_b_disk_i))
            sign_B_flux.append(literal_eval(sign_b_flux_i))
            sign_B_jet.append(literal_eval(sign_b_jet_i))
            sign_B_upper.append(literal_eval(sign_b_up_i))
            sign_B_lower.append(literal_eval(sign_b_low_i))
            sign_B_disk.append(literal_eval(sign_b_disk_i))

    time, abs_B_flux, abs_B_jet, abs_B_upper, abs_B_lower, abs_B_disk, sign_B_flux, sign_B_jet, sign_B_upper, sign_B_lower, sign_B_disk = zip(*sorted(zip(time, abs_B_flux, abs_B_jet, abs_B_upper, abs_B_lower, abs_B_disk, sign_B_flux, sign_B_jet, sign_B_upper, sign_B_lower, sign_B_disk)))

    time = np.asarray(time)
    mask = time > 0.5e5

    abs_B_flux = np.asarray(abs_B_flux)
    abs_B_jet = np.asarray(abs_B_jet)
    abs_B_upper = np.asarray(abs_B_upper)
    abs_B_lower = np.asarray(abs_B_lower)
    abs_B_disk = np.asarray(abs_B_disk)
    sign_B_flux = np.asarray(sign_B_flux)
    sign_B_jet = np.asarray(sign_B_jet)
    sign_B_upper = np.asarray(sign_B_upper)
    sign_B_lower = np.asarray(sign_B_lower)
    sign_B_disk = np.asarray(sign_B_disk)

    #plot_signed(time, sign_B_flux, sign_B_jet, sign_B_upper, sign_B_lower, sign_B_disk, data_dir)
    #plot_all(time, abs_B_flux, abs_B_jet, abs_B_upper, abs_B_lower, abs_B_disk, sign_B_flux, sign_B_jet, sign_B_upper, sign_B_lower, sign_B_disk, data_dir)
    #plot_stacked(time, abs_B_flux, abs_B_jet, abs_B_upper, abs_B_lower, abs_B_disk, data_dir)
    plot_ratio(time[mask], abs_B_flux[mask], sign_B_flux[mask], data_dir)

def plot_ratio(time, a_B_f, s_B_f, data_dir):
    R_u = s_B_f[:,0]/a_B_f[:,0]
    R_l = s_B_f[:,1]/a_B_f[:,1]

    fig, axs = plt.subplots(2, 1, figsize=(5,5), sharex=True, sharey=True)
    axs[0].plot(time, R_u, linewidth=1, color='k')
    axs[0].set_xlabel(r'time ($GM/c^3$)')
    axs[0].set_ylabel(r'$\frac{\int B_r \cdot dS}{\int|{B_r}|\cdot dS}$')
    axs[0].set_title('Upper hemisphere')
    axs[0].set_ylim(-1,1)

    axs[1].plot(time, R_l, linewidth=1, color='k')
    axs[1].set_xlabel(r'time ($GM/c^3$)')
    axs[1].set_ylabel(r'$\frac{\int B_r \cdot dS}{\int|{B_r}|\cdot dS}$')
    axs[1].set_title('Lower hemisphere')
    axs[1].set_ylim(-1,1)


    for ax in fig.get_axes():
        #ax.set_yscale("log", base=10)
        ax.label_outer()
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        ax.minorticks_on()
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig(data_dir + 'B_ratio_' + args.component + '.png', dpi=1200)
    plt.close()

def plot_stacked(time, B_f, B_j, B_u, B_l, B_d, data_dir):
    upper_B_by_type = {
    'disk': B_d[:,1],
    'lower_atmos': B_l[:,1],
    'upper_atmos': B_u[:,1],
    'jet': B_j[:,1],
    }

    lower_B_by_type = {
    'disk': B_d[:,0],
    'lower_atmos': B_l[:,0],
    'upper_atmos': B_u[:,0],
    'jet': B_j[:,0],
    }

    fig, axs = plt.subplots(2, 1, figsize=(5,5), sharex=True, sharey=True)
    axs[0].stackplot(time, upper_B_by_type.values(), labels=upper_B_by_type.keys())
    axs[0].legend(loc='upper left')
    axs[0].set_title('Upper')
    axs[0].set_xlabel('time (GM/c3)')
    axs[0].set_ylabel('average B')

    axs[1].stackplot(time, lower_B_by_type.values(), labels=lower_B_by_type.keys())
    axs[1].legend(loc='upper left')
    axs[1].set_title('Lower')
    axs[1].set_xlabel('time (GM/c3)')
    axs[1].set_ylabel('average B')

    for ax in fig.get_axes():
        ax.label_outer()
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        ax.minorticks_on()
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.tight_layout()
    plt.savefig(data_dir + 'B_strength_' + args.component + '_stacked.png', dpi=1200)
    #plt.show()

def plot_signed(time, B_f, B_j, B_u, B_l, B_d, data_dir):
    fig, axs = plt.subplots(2, 1, figsize=(5,5), sharex=True, sharey=True)
    labels = ['jet', 'upper atmos.', 'lower atmos.', 'in disk', 'from flux']
    lw = 1
    a = 0.5

    axs[0].plot(time, B_j[:,1], linewidth=lw, alpha=a)
    axs[0].plot(time, B_u[:,1], linewidth=lw, alpha=a)
    axs[0].plot(time, B_l[:,1], linewidth=lw, alpha=a)
    axs[0].plot(time, B_d[:,1], linewidth=lw, alpha=a)
    axs[0].plot(time, B_f[:,0], linewidth=1, color='k') #for some reason, Bflux is reversed
    axs[0].set_xlabel(r'time ($GM/c^3$)')
    axs[0].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (upper)')
    axs[0].set_title('Signed')

    axs[1].plot(time, B_j[:,0], linewidth=lw, alpha=a)
    axs[1].plot(time, B_u[:,0], linewidth=lw, alpha=a)
    axs[1].plot(time, B_l[:,0], linewidth=lw, alpha=a)
    axs[1].plot(time, B_d[:,0], linewidth=lw, alpha=a)
    axs[1].plot(time, B_f[:,1], linewidth=1, color='k')
    axs[1].set_xlabel(r'time ($GM/c^3$)')
    axs[1].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (lower)')

    for ax in fig.get_axes():
        ax.label_outer()
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        ax.minorticks_on()
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    lg = plt.legend(labels,bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    plt.savefig(data_dir + 'B_strength_' + args.component + '_signed.png', dpi=1200, bbox_extra_artists=(lg,), bbox_inches='tight')
    plt.close()

def plot_all(time, a_B_f, a_B_j, a_B_u, a_B_l, a_B_d, s_B_f, s_B_j, s_B_u, s_B_l, s_B_d, data_dir):
    fig, axs = plt.subplots(2, 2, figsize=(8,5), sharex=True, sharey=True)
    labels = ['jet', 'upper atmos.', 'lower atmos.', 'in disk', 'from flux']
    lw = 1
    a = 0.5

    axs[0,0].plot(time, a_B_j[:,1], linewidth=lw, alpha=a)
    axs[0,0].plot(time, a_B_u[:,1], linewidth=lw, alpha=a)
    axs[0,0].plot(time, a_B_l[:,1], linewidth=lw, alpha=a)
    axs[0,0].plot(time, a_B_d[:,1], linewidth=lw, alpha=a)
    axs[0,0].plot(time, a_B_f[:,1], linewidth=1, color='k')
    axs[0,0].set_xlabel(r'time ($GM/c^3$)')
    axs[0,0].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (upper)')
    axs[0,0].set_title('Average')

    axs[0,1].plot(time, s_B_j[:,1], linewidth=lw, alpha=a)
    axs[0,1].plot(time, s_B_u[:,1], linewidth=lw, alpha=a)
    axs[0,1].plot(time, s_B_l[:,1], linewidth=lw, alpha=a)
    axs[0,1].plot(time, s_B_d[:,1], linewidth=lw, alpha=a)
    axs[0,1].plot(time, s_B_f[:,0], linewidth=1, color='k')
    axs[0,1].set_xlabel(r'time ($GM/c^3$)')
    axs[0,1].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (upper)')
    axs[0,1].set_title('Signed')

    axs[1,0].plot(time, a_B_j[:,0], linewidth=lw, alpha=a)
    axs[1,0].plot(time, a_B_u[:,0], linewidth=lw, alpha=a)
    axs[1,0].plot(time, a_B_l[:,0], linewidth=lw, alpha=a)
    axs[1,0].plot(time, a_B_d[:,0], linewidth=lw, alpha=a)
    axs[1,0].plot(time, a_B_f[:,0], linewidth=1, color='k')
    axs[1,0].set_xlabel(r'time ($GM/c^3$)')
    axs[1,0].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (lower)')

    axs[1,1].plot(time, s_B_j[:,0], linewidth=lw, alpha=a)
    axs[1,1].plot(time, s_B_u[:,0], linewidth=lw, alpha=a)
    axs[1,1].plot(time, s_B_l[:,0], linewidth=lw, alpha=a)
    axs[1,1].plot(time, s_B_d[:,0], linewidth=lw, alpha=a)
    axs[1,1].plot(time, s_B_f[:,1], linewidth=1, color='k')
    axs[1,1].set_xlabel(r'time ($GM/c^3$)')
    axs[1,1].set_ylabel(r'$\overline{B}_{r\sim15r_g}$ (lower)')

    for ax in fig.get_axes():
        ax.label_outer()
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        ax.minorticks_on()
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend(labels,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(data_dir + 'B_strength_' + args.component + '.png', dpi=1200)
    plt.close()

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
    parser.add_argument('-c', '--component',
                        type=str,
                        default='all',
                        help='specify the B component to calculate, e.g. r, theta, phi, all (default:all)')
    args = parser.parse_args()

    main(**vars(args))
