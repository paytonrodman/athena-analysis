#!/usr/bin/env python3
#
# calc_svd.py
#
# A program to calculate the effective EMF and average magnetic field for use in SVD calculations,
# using MPI.
#
# To run:
# mpirun -n [n] python calc_svd.py [options]
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
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.linalg import svd

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

    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + 'beta_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time = np.append(csv_time, float(row[0]))

    files = glob.glob('./high_res.cons.*.athdf')
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

    local_times = [10000,15000,20000]

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
    th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

    del data_input
    del data_init

    if rank==0:
        if not args.update:
            with open(prob_dir + 'beta_with_time.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "plasma_beta"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + ".cons." + str_t + ".athdf", quantities=['mom1','mom2','mom3','Bcc1','Bcc2','Bcc3'])

        #unpack data
        #x1f = data_cons['x1f'] # r
        #x2f = data_cons['x2f'] # theta
        #x3f = data_cons['x3f'] # phi
        #x1v = data_cons['x1v'] # r
        #x2v = data_cons['x2v'] # theta
        #x3v = data_cons['x3v'] # phi
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        # azimuthal average of each component, expanded out to 3D again
        mom1_av = np.array([np.average(mom1,axis=0)]*np.shape(mom1)[0])
        mom2_av = np.array([np.average(mom2,axis=0)]*np.shape(mom2)[0])
        mom3_av = np.array([np.average(mom3,axis=0)]*np.shape(mom3)[0])
        Bcc1_av = np.array([np.average(Bcc1,axis=0)]*np.shape(Bcc1)[0])
        Bcc2_av = np.array([np.average(Bcc2,axis=0)]*np.shape(Bcc2)[0])
        Bcc3_av = np.array([np.average(Bcc3,axis=0)]*np.shape(Bcc3)[0])

        # fluctuating value of each component
        mom1_fluc = mom1 - mom1_av
        mom2_fluc = mom2 - mom2_av
        mom3_fluc = mom3 - mom3_av
        Bcc1_fluc = Bcc1 - Bcc1_av
        Bcc2_fluc = Bcc2 - Bcc2_av
        Bcc3_fluc = Bcc3 - Bcc3_av

        emf1 = mom2_fluc*Bcc3_fluc - mom3_fluc*Bcc2_fluc # vth*Bph - vph*Bth
        emf2 = - (mom1_fluc*Bcc3_fluc - mom3_fluc*Bcc1_fluc) # - (vr*Bph - vph*Br)
        emf3 = mom1_fluc*Bcc2_fluc - mom2_fluc*Bcc1_fluc #vr*Bth - vth*Br

        


        #mom_all = np.sqrt(np.square(mom1) + np.square(mom2) + np.square(mom3))
        #mom_all_av = np.sqrt(np.square(mom1_av) + np.square(mom2_av) + np.square(mom3_av))
        #mom_all_av = np.array([mom_all_av]*np.shape(mom1)[0])
        #mom_fluc = mom_all_av - mom_all
        #mom_fluc_av = np.average(mom_fluc,axis=0)

        # Find volume centred total magnetic field
        #bcc_all = np.sqrt(np.square(Bcc1) + np.square(Bcc2) + np.square(Bcc3))
        #bcc_all_av = np.sqrt(np.square(Bcc1_av) + np.square(Bcc2_av) + np.square(Bcc3_av))
        #bcc_all_av = np.array([bcc_all_av]*np.shape(Bcc1)[0])
        #bcc_fluc = bcc_all_av - bcc_all
        #bcc_fluc_av = np.average(bcc_fluc,axis=0)



        plt.imshow(bcc_fluc_av)
        plt.show()

        print(qwjbjw)

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + 'beta_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,current_beta]
            writer.writerow(row)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart (reduces computational time by only appending to files, rather than rewriting)')
    args = parser.parse_args()

    main(**vars(args))
