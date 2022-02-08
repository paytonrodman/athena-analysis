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
from mpi4py import MPI

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    if args.average not in ['azimuthal','gaussian']:
        sys.exit('Must select averaging method (azimuthal, gaussian)')

    if args.hemisphere not in ['upper','lower']:
        sys.exit('Must select hemisphere (upper, lower)')

    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + 'dyn_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time = np.append(csv_time, float(row[0]))

    files = glob.glob('./'+problem+'.cons.*.athdf')
    times = np.empty(0)
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) not in csv_time:
                times = np.append(times, float(time_sec[0]))
        else:
            if float(time_sec[0]) not in times:
                times = np.append(times, float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    # distribute files to cores
    count = len(times) // size  # number of files for each process to analyze
    remainder = len(times) % size  # extra files if times is not a multiple of size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    #local_times = [1500,3580] #43159 51273

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    scale_height = data_input['problem']['h_r']
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max'] # bounds of high resolution region
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max'] # bounds of high resolution region
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max'] # bounds of high resolution region
    else:
        x1_high_max = data_input['mesh']['x1max']
    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r_u = AAT.find_nearest(x1v, x1_high_max)
    if args.hemisphere=='upper':
        th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))
        th_u = AAT.find_nearest(x2v, np.pi/2. - (1.*scale_height))
    elif args.hemisphere=="lower":
        th_l = AAT.find_nearest(x2v, np.pi/2. + (1.*scale_height))
        th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))

    if rank==0:
        if not args.update:
            with open(prob_dir + 'dyn_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "alpha", "C"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['mom1','mom2','mom3','Bcc1','Bcc2','Bcc3'])

        #unpack data
        mom1 = data_cons['mom1'][:,th_l:th_u,:r_u] #select points between 1-2 scale heights
        mom2 = data_cons['mom2'][:,th_l:th_u,:r_u]
        mom3 = data_cons['mom3'][:,th_l:th_u,:r_u]
        Bcc1 = data_cons['Bcc1'][:,th_l:th_u,:r_u]
        Bcc2 = data_cons['Bcc2'][:,th_l:th_u,:r_u]
        Bcc3 = data_cons['Bcc3'][:,th_l:th_u,:r_u]

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

        alpha1_i, C1_i = np.polyfit(emf1_av.flatten(), Bcc1_av.flatten(), 1)
        alpha2_i, C2_i = np.polyfit(emf2_av.flatten(), Bcc2_av.flatten(), 1)
        alpha3_i, C3_i = np.polyfit(emf3_av.flatten(), Bcc3_av.flatten(), 1)
        alpha = [alpha1_i,alpha2_i,alpha3_i]
        C = [C1_i,C2_i,C3_i]

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + 'dyn_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,alpha,C])


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
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
