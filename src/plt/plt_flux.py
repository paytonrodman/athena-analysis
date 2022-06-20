#!/usr/bin/env python3
#
# plt_flux.py
#
# A program to plot the time series of the magnetic flux crossing the ISCO, from data generated
# by calc_flux.py
#
# Usage: python plt_flux.py [options]
#
# Python standard modules
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')

# Other Python modules
from ast import literal_eval
import csv
import matplotlib.pyplot as plt
import numpy as np

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    data_dir = root_dir + problem + '/'
    os.chdir(data_dir)

    if args.prob_id=='high_res':
        min_time = 2.5e4
    elif args.prob_id=='high_beta' or args.prob_id=='super_res':
        min_time = 1e4

    time = []
    mag_flux = []
    mag_flux_abs = []
    with open('flux_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            mf_u = float(row[2])
            mf_l = float(row[3])
            mf_u_a = float(row[4])
            mf_l_a = float(row[5])
            time.append(t)
            mag_flux.append([mf_u,mf_l])
            mag_flux_abs.append([mf_u_a,mf_l_a])

    time_add = []
    mass_flux = []
    with open('mass_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            mass_i = literal_eval(row[2])
            time_add.append(t)
            mass_flux.append(mass_i[0])

    time, mag_flux, mag_flux_abs = zip(*sorted(zip(time, mag_flux, mag_flux_abs)))
    time_add, mass_flux = zip(*sorted(zip(time_add, mass_flux)))

    if args.mag:
        data = mag_flux_abs
        ylab1 = r'$\sqrt{4\pi} \oint_{r_{\rm{hor}}} |B_r| \cdot dS$'
        savename = data_dir + 'mag_flux_abs.png'
    elif args.ratio:
        data = np.asarray(mag_flux) / np.asarray(mag_flux_abs)
        ylab1 = r'$\frac{\oint B_r \cdot dS}{\oint |B_r| \cdot dS}$'
        savename = data_dir + 'mag_flux_ratio.png'
    else:
        data = mag_flux
        ylab1 = r'$\sqrt{4\pi} \oint_{r_{\rm{hor}}} B_r \cdot dS$'
        savename = data_dir + 'mag_flux.png'

    if not args.ratio:
        data_average = [(x[0]+x[1])/2. for x in data]
    else:
        data_average = [(np.abs(x[0])+np.abs(x[1]))/2. for x in data]

    if args.av_mass:
        time_add = np.array(time_add)
        mass_flux = np.array(mass_flux)
        mass_average = np.average(mass_flux[time_add > min_time])
        rawtoscaled = 1./(np.sqrt(mass_average))
        ylab2 = r'$\Phi / \sqrt{\langle\dot{M}\rangle}$'
    else:
        with np.errstate(divide='ignore',invalid='ignore'):
            if args.mag:
                data_scaled = [data_average[ii]/(np.sqrt(mass_flux[ii])) for ii,d in enumerate(data_average)]
                ylab1 = r'$\sqrt{4\pi} \oint_{r_{\rm{hor}}} |B_r| \cdot dS / \sqrt{\dot{M}}$'
            else:
                data_scaled = [data[ii]/(np.sqrt(mass_flux[ii])) for ii,d in enumerate(data)]
                ylab1 = r'$\sqrt{4\pi} \oint_{r_{\rm{hor}}} B_r \cdot dS / \sqrt{\dot{M}}$'

    _, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    lw=1.5

    if args.ratio:
        ax1.plot(time, data_average, linewidth=lw)
        ax1.set_ylim(top=1,bottom=0)
    else:
        if args.av_mass:
            if args.mag:
                ax1.plot(time, data_average, linewidth=lw)
                ax1.set_ylim(bottom=0)
            else:
                ax1.axhline(y=0, color="grey",alpha=0.5)
                ax1.plot(time, [d[0] for d in data], linewidth=lw)
                ax1.plot(time, [d[1] for d in data], linewidth=lw)
            ax2 = ax1.twinx()
            mn, mx = ax1.get_ylim()
            min_scale = mn*rawtoscaled
            max_scale = mx*rawtoscaled
            ax2.set_ylim(min_scale, max_scale)
            ax2.set_ylabel(ylab2)
        else:
            if args.mag:
                ax1.plot(time, data_scaled, linewidth=lw)
                ax1.set_ylim(bottom=0)
            else:
                ax1.axhline(y=0, color="grey",alpha=0.5)
                ax1.plot(time, [d[0] for d in data_scaled], linewidth=lw)
                ax1.plot(time, [d[1] for d in data_scaled], linewidth=lw)
            #ax2 = ax1.twinx()
            #mn, mx = ax1.get_ylim()
            #min = np.nanmin(data_scaled)
            #max = np.nanmax(data_scaled)



    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    ax1.set_xlabel(r'time ($GM/c^3$)')
    ax1.set_ylabel(ylab1)
    ax1.set_xlim(left=0)

    if args.grid:
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.3)
        ax1.minorticks_on()
        ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    plt.savefig(savename)
    plt.close()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot magnetic flux across both hemispheres over time.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('--ratio',
                        action='store_true',
                        help='whether to plot ratio of abs to non-abs')
    parser.add_argument('--grid',
                        action='store_true',
                        help='plot grid')
    parser.add_argument('--mag',
                        action='store_true',
                        help='plot integral of magnitude of B')
    parser.add_argument('--av_mass',
                        action='store_true',
                        help='use the time-averaged mass to scale the flux')
    args = parser.parse_args()

    main(**vars(args))
