#!/usr/bin/env python3
#
# calc_dyn.py
#
# A program to calculate the diagonal dynamo coefficients through the turbulent EMF
#
# Usage: mpirun -n [nprocs] calc_dyn.py [options]
#
# Python standard modules
import argparse
import sys
import os
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
import csv

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(args.data)

    file_times = AAT.add_time_to_list(args.update, args.output)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    local_times = [25000]

    data_input = athena_read.athinput(args.input)
    scale_height = data_input['problem']['h_r']
    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x1v','x2v','x3v'])
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
    elif args.hemisphere=='lower':
        th_l = AAT.find_nearest(x2v, np.pi/2. + (np.arctan(1.*dist/x1v[r_mid])))
        th_u = AAT.find_nearest(x2v, np.pi/2. + (np.arctan(2.*dist/x1v[r_mid])))
    # analyze phi wedge instead of full disk
    if args.reduced:
        ph_l = AAT.find_nearest(x3v, x3v[0])
        ph_u = AAT.find_nearest(x3v, x3v[int(np.size(x3v)/4.)])
    else:
        ph_l = AAT.find_nearest(x3v, x3v[0])
        ph_u = AAT.find_nearest(x3v, x3v[-1])

    if rank==0:
        if not args.update:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['sim_time', 'orbit_time', 'alpha', 'C'])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['mom1','mom2','mom3','dens','Bcc1','Bcc2','Bcc3'])

        # unpack data and select points between 1-2 scale heights
        mom1 = data_cons['mom1'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        mom2 = data_cons['mom2'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        mom3 = data_cons['mom3'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        dens = data_cons['dens'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        v1 = mom1/dens
        v2 = mom2/dens
        v3 = mom3/dens
        Bcc1 = data_cons['Bcc1'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        Bcc2 = data_cons['Bcc2'][ph_l:ph_u,th_l:th_u,r_l:r_u]
        Bcc3 = data_cons['Bcc3'][ph_l:ph_u,th_l:th_u,r_l:r_u]

        if args.average=='gaussian':
            s=5
            v1_av = gaussian_filter(v1, sigma=s)
            v2_av = gaussian_filter(v2, sigma=s)
            v3_av = gaussian_filter(v3, sigma=s)
            Bcc1_av = gaussian_filter(Bcc1, sigma=s)
            Bcc2_av = gaussian_filter(Bcc2, sigma=s)
            Bcc3_av = gaussian_filter(Bcc3, sigma=s)
        elif args.average=='azimuthal':
            v1_av = np.average(v1, axis=0)
            v2_av = np.average(v2, axis=0)
            v3_av = np.average(v3, axis=0)
            Bcc1_av = np.average(Bcc1, axis=0)
            Bcc2_av = np.average(Bcc2, axis=0)
            Bcc3_av = np.average(Bcc3, axis=0)
            # expand back out to original mesh
            v1_av = np.repeat(v1_av[np.newaxis, :, :], np.shape(v1)[0], axis=0)
            v2_av = np.repeat(v2_av[np.newaxis, :, :], np.shape(v2)[0], axis=0)
            v3_av = np.repeat(v3_av[np.newaxis, :, :], np.shape(v3)[0], axis=0)
            Bcc1_av = np.repeat(Bcc1_av[np.newaxis, :, :], np.shape(Bcc1)[0], axis=0)
            Bcc2_av = np.repeat(Bcc2_av[np.newaxis, :, :], np.shape(Bcc2)[0], axis=0)
            Bcc3_av = np.repeat(Bcc3_av[np.newaxis, :, :], np.shape(Bcc3)[0], axis=0)


        # fluctuating components of velocity and magnetic field
        v1_fluc = v1 - v1_av
        v2_fluc = v2 - v2_av
        v3_fluc = v3 - v3_av
        Bcc1_fluc = Bcc1 - Bcc1_av
        Bcc2_fluc = Bcc2 - Bcc2_av
        Bcc3_fluc = Bcc3 - Bcc3_av

        # EMF components from cross product components
        emf1 = v2_fluc*Bcc3_fluc - v3_fluc*Bcc2_fluc
        emf2 = -(v1_fluc*Bcc3_fluc - v3_fluc*Bcc1_fluc)
        emf3 = v1_fluc*Bcc2_fluc - v2_fluc*Bcc1_fluc

        if args.average=='gaussian':
            emf1_av = gaussian_filter(emf1, sigma=s)
            emf2_av = gaussian_filter(emf2, sigma=s)
            emf3_av = gaussian_filter(emf3, sigma=s)
        elif args.average=='azimuthal':
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

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,alpha,C])


def func(x, a, b):
    y = a*x + b
    return y


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        help='location of athinput file, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    parser.add_argument('--reduced',
                        action='store_true',
                        help='average over a reduced azimuthal range')
    parser.add_argument('-a', '--average',
                        type=str,
                        default='azimuthal',
                        choices=['azimuthal', 'gaussian'],
                        help='averaging method')
    parser.add_argument('-H', '--hemisphere',
                        type=str,
                        default='upper',
                        choices=['upper', 'lower'],
                        help='hemisphere to average in')
    parser.add_argument('--r',
                        type=float,
                        default=None,
                        help=('value of r where averages are calculated (default: 25rg)'))
    args = parser.parse_args()

    main(**vars(args))
