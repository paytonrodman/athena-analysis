#!/usr/bin/env python3
#
# calc_dyn.py
#
# A program to calculate the diagonal dynamo coefficients through the turbulent EMF
#
# To run:
# mpirun -n [n] python calc_dyn.py [options]
# for [n] cores.
#
import numpy as np
import os
import sys
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
from math import sqrt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import pandas as pd
#from sklearn.linear_model import HuberRegressor
from scipy import optimize

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    av = args.average
    hem = args.hemisphere

    if args.average not in ['azimuthal','gaussian']:
        sys.exit('Must select averaging method (azimuthal, gaussian)')
    if args.hemisphere not in ['upper','lower']:
        sys.exit('Must select hemisphere (upper, lower)')

    files = glob.glob('./'+problem+'.cons.*.athdf')
    times = np.empty(0)
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if float(time_sec[0]) not in times:
            times = np.append(times, float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    times = [6140] #43159 51273 6140 5000

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    scale_height = data_input['problem']['h_r']
    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r_l = AAT.find_nearest(x1v, 15.)
    r_u = AAT.find_nearest(x1v, 15. + 15.*scale_height)
    r_mid = np.int(r_l + (r_u - r_l)/2.)
    dist = scale_height*x1v[r_mid]
    if args.hemisphere=='upper':
        th_l = AAT.find_nearest(x2v, np.pi/2. - (np.arctan(2.*dist/x1v[r_mid])))
        th_u = AAT.find_nearest(x2v, np.pi/2. - (np.arctan(1.*dist/x1v[r_mid])))
    elif args.hemisphere=="lower":
        th_l = AAT.find_nearest(x2v, np.pi/2. + (np.arctan(1.*dist/x1v[r_mid])))
        th_u = AAT.find_nearest(x2v, np.pi/2. + (np.arctan(2.*dist/x1v[r_mid])))

    for t in times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['mom1','mom2','mom3','Bcc1','Bcc2','Bcc3'])

        #unpack data
        mom1 = data_cons['mom1'][:,th_l:th_u,r_l:r_u] #select points between 1-2 scale heights
        mom2 = data_cons['mom2'][:,th_l:th_u,r_l:r_u]
        mom3 = data_cons['mom3'][:,th_l:th_u,r_l:r_u]
        Bcc1 = data_cons['Bcc1'][:,th_l:th_u,r_l:r_u]
        Bcc2 = data_cons['Bcc2'][:,th_l:th_u,r_l:r_u]
        Bcc3 = data_cons['Bcc3'][:,th_l:th_u,r_l:r_u]

        if args.average=='gaussian':
            s=5
            mom1_av = gaussian_filter(mom1, sigma=s)
            mom2_av = gaussian_filter(mom2, sigma=s)
            mom3_av = gaussian_filter(mom3, sigma=s)
            Bcc1_av = gaussian_filter(Bcc1, sigma=s)
            Bcc2_av = gaussian_filter(Bcc2, sigma=s)
            Bcc3_av = gaussian_filter(Bcc3, sigma=s)

        if args.average=='azimuthal':
            mom1_av = np.average(mom1, axis=0)
            mom2_av = np.average(mom2, axis=0)
            mom3_av = np.average(mom3, axis=0)
            Bcc1_av = np.average(Bcc1, axis=0)
            Bcc2_av = np.average(Bcc2, axis=0)
            Bcc3_av = np.average(Bcc3, axis=0)
            mom1_av = np.repeat(mom1_av[np.newaxis, :, :], np.shape(mom1)[0], axis=0)
            mom2_av = np.repeat(mom2_av[np.newaxis, :, :], np.shape(mom2)[0], axis=0)
            mom3_av = np.repeat(mom3_av[np.newaxis, :, :], np.shape(mom3)[0], axis=0)
            Bcc1_av = np.repeat(Bcc1_av[np.newaxis, :, :], np.shape(Bcc1)[0], axis=0)
            Bcc2_av = np.repeat(Bcc2_av[np.newaxis, :, :], np.shape(Bcc2)[0], axis=0)
            Bcc3_av = np.repeat(Bcc3_av[np.newaxis, :, :], np.shape(Bcc3)[0], axis=0)

        # fluctuating components of momentum and magnetic field
        mom1_fluc = mom1 - mom1_av
        mom2_fluc = mom2 - mom2_av
        mom3_fluc = mom3 - mom3_av
        Bcc1_fluc = Bcc1 - Bcc1_av
        Bcc2_fluc = Bcc2 - Bcc2_av
        Bcc3_fluc = Bcc3 - Bcc3_av

        # EMF components from cross product components
        emf1 = mom2_fluc*Bcc3_fluc - mom3_fluc*Bcc2_fluc
        emf2 = -(mom1_fluc*Bcc3_fluc - mom3_fluc*Bcc1_fluc)
        emf3 = mom1_fluc*Bcc2_fluc - mom2_fluc*Bcc1_fluc

        if args.average=='gaussian':
            emf1_av = gaussian_filter(emf1, sigma=s)
            emf2_av = gaussian_filter(emf2, sigma=s)
            emf3_av = gaussian_filter(emf3, sigma=s)

        if args.average=='azimuthal':
            emf1_av = np.average(emf1, axis=0)
            emf2_av = np.average(emf2, axis=0)
            emf3_av = np.average(emf3, axis=0)
            emf1_av = np.repeat(emf1_av[np.newaxis, :, :], np.shape(emf1)[0], axis=0)
            emf2_av = np.repeat(emf2_av[np.newaxis, :, :], np.shape(emf2)[0], axis=0)
            emf3_av = np.repeat(emf3_av[np.newaxis, :, :], np.shape(emf3)[0], axis=0)

        emf_all = [emf1_av,emf2_av,emf3_av]
        Bcc_all = [Bcc1_av,Bcc2_av,Bcc3_av]

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        for num in range(0,len(emf_all)):
            emf = emf_all[num]
            Bcc = Bcc_all[num]
            if num==0:
                xlab1 = r'$\overline{B}_{r}$'
                ylab1 = r'$\epsilon_{r}$'
                f3 = 'r'
            elif num==1:
                xlab1 = r'$\overline{B}_{\theta}$'
                ylab1 = r'$\epsilon_{\theta}$'
                f3 = 'th'
            elif num==2:
                xlab1 = r'$\overline{B}_{\phi}$'
                ylab1 = r'$\epsilon_{\phi}$'
                f3 = 'ph'

            x = Bcc.flatten()
            y = emf.flatten()

            # HUBER REGRESSION
            #huber = HuberRegressor()
            #huber.fit(x.reshape(-1,1), y)
            #x_fit = np.linspace(np.min(x),np.max(x),100)
            #y_fit = huber.coef_*x_fit + huber.intercept_

            # LINEAR LEAST SQUARES
            alpha,C = optimize.curve_fit(func, xdata=x, ydata=y)[0]
            y_fit = alpha*x + C
            x_fit = x

            #idx = [randint(0, np.size(x)-1) for p in range(0, 10000)]
            #x = x[idx]
            #y = y[idx]

            df = pd.DataFrame(list(zip(x, y)), columns =['Bcc', 'emf'])
            sns.set_style("ticks")
            sns.despine()
            ax = sns.kdeplot(x=df.Bcc, y=df.emf, cmap="Reds", shade=True, bw_adjust=5)
            ax.set(xlabel=xlab1, ylabel=ylab1)
            title_str = "t=" + str(int(sim_t)) + " (" + str(int(orbit_t)) + " orbits)"
            ax.set_title(title_str)

            plt.plot(x_fit, y_fit, '-k', linewidth=1)
            #plt.xlim([-0.002, 0.002])
            #plt.ylim([-6e-6, 6e-6])

            plt.tight_layout()
            plt.savefig(prob_dir+'dyn/'+av[:3]+'_'+hem+'_'+f3+'_'+str_t+'.png',dpi=300)
            plt.close()
            #plt.show()

def func(x, a, b):
    y = a*x + b
    return y


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-a', '--average',
                        type=str,
                        default='azimuthal',
                        help='specify averaging method (azimuthal,gaussian)')
    parser.add_argument('-H', '--hemisphere',
                        type=str,
                        default='upper',
                        help='specify which hemisphere to average in (upper, lower)')
    parser.add_argument('--r',
                        type=float,
                        default=None,
                        help=('value of r where averages are calculated (default: 25rg)'))
    args = parser.parse_args()

    main(**vars(args))
