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
import os
import sys
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
#import glob
#import re
import csv
import argparse
import numpy as np
from math import sqrt
from scipy import optimize
from scipy.ndimage import gaussian_filter
from mpi4py import MPI

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + args.prob_id + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    filename_output = 'dyn/huber/' + args.average[:3] + '_' + args.hemisphere + '_dyn_with_time.csv'
    os.chdir(data_dir)

    if args.average not in ['azimuthal','gaussian']:
        sys.exit('Must select averaging method (azimuthal, gaussian)')
    if args.hemisphere not in ['upper','lower']:
        sys.exit('Must select hemisphere (upper, lower)')

    file_times = AAT.add_time_to_list(args.update, prob_dir, filename_output, args.prob_id)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + args.prob_id)
    scale_height = data_input['problem']['h_r']
    data_init = athena_read.athdf(args.prob_id + '.cons.00000.athdf', quantities=['x1v','x2v','x3v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    x3v = data_init['x3v']
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
    if args.reduced:
        ph_l = AAT.find_nearest(x3v, x3v[0])
        ph_u = AAT.find_nearest(x3v, x3v[int(np.size(x3v)/4.)])
    else:
        ph_l = AAT.find_nearest(x3v, x3v[0])
        ph_u = AAT.find_nearest(x3v, x3v[-1])

    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "alpha", "C"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.prob_id + '.cons.' + str_t + '.athdf', quantities=['mom1','mom2','mom3','Bcc1','Bcc2','Bcc3'])

        #unpack data and select points between 1-2 scale heights
        mom1 = data_cons['mom1'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        mom2 = data_cons['mom2'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        mom3 = data_cons['mom3'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        Bcc1 = data_cons['Bcc1'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        Bcc2 = data_cons['Bcc2'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        Bcc3 = data_cons['Bcc3'][ph_l:ph_u,th_l:th_u,r_l:r_u]

        if args.average=='gaussian':
            s=5
            mom1_av = gaussian_filter(mom1, sigma=s)
            mom2_av = gaussian_filter(mom2, sigma=s)
            mom3_av = gaussian_filter(mom3, sigma=s)
            Bcc1_av = gaussian_filter(Bcc1, sigma=s)
            Bcc2_av = gaussian_filter(Bcc2, sigma=s)
            Bcc3_av = gaussian_filter(Bcc3, sigma=s)

        if args.average=='azimuthal':
            mom1_av = np.average(mom1[ph_l:ph_u,:,:], axis=0)
            mom2_av = np.average(mom2[ph_l:ph_u,:,:], axis=0)
            mom3_av = np.average(mom3[ph_l:ph_u,:,:], axis=0)
            Bcc1_av = np.average(Bcc1[ph_l:ph_u,:,:], axis=0)
            Bcc2_av = np.average(Bcc2[ph_l:ph_u,:,:], axis=0)
            Bcc3_av = np.average(Bcc3[ph_l:ph_u,:,:], axis=0)
            mom1_av = np.repeat(mom1_av[np.newaxis, :, :], np.shape(mom1[ph_l:ph_u,:,:])[0], axis=0)
            mom2_av = np.repeat(mom2_av[np.newaxis, :, :], np.shape(mom2[ph_l:ph_u,:,:])[0], axis=0)
            mom3_av = np.repeat(mom3_av[np.newaxis, :, :], np.shape(mom3[ph_l:ph_u,:,:])[0], axis=0)
            Bcc1_av = np.repeat(Bcc1_av[np.newaxis, :, :], np.shape(Bcc1[ph_l:ph_u,:,:])[0], axis=0)
            Bcc2_av = np.repeat(Bcc2_av[np.newaxis, :, :], np.shape(Bcc2[ph_l:ph_u,:,:])[0], axis=0)
            Bcc3_av = np.repeat(Bcc3_av[np.newaxis, :, :], np.shape(Bcc3[ph_l:ph_u,:,:])[0], axis=0)


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

        B_all = [Bcc1_av,Bcc2_av,Bcc3_av]
        emf_all = [emf1_av,emf2_av,emf3_av]

        alpha = []
        C = []
        for num in range(0,3):
            x = B_all[num]
            y = emf_all[num]

            if t==0:
                alpha_i,C_i = np.nan, np.nan
            else:
                alpha_i,C_i = optimize.curve_fit(func, xdata=x.flatten(), ydata=y.flatten())[0]
            alpha.append(alpha_i)
            C.append(C_i)

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,alpha,C])

def func(x, a, b):
    y = a*x + b
    return y


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('--reduced',
                        action="store_true",
                        help='specify whether to average over a reduced azimuthal range')
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
