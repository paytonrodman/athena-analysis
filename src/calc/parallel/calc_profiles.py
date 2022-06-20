#!/usr/bin/env python3
#
# calc_beta.py
#
# A program to calculate the plasma beta of an Athena++ disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_beta.py [options]
#
# Python standard modules
import argparse
import sys
import os
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np
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

    os.chdir(kwargs['data'])

    file_times = AAT.add_time_to_list(kwargs['update'], kwargs['output'])
    if kwargs['problem_id']=='high_res':
        file_times = file_times[file_times>10000] # t > 5e4
    elif kwargs['problem_id']=='high_beta':
        file_times = file_times[file_times>4000] # t > 2e4
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(kwargs['input'])
    scale_height = data_input['problem']['h_r']
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max'] # bounds of high resolution region
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max'] # bounds of high resolution region
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max'] # bounds of high resolution region
    else:
        x1_high_max = data_input['mesh']['x1max']

    data_init = athena_read.athdf(kwargs['problem_id'] + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r_u = AAT.find_nearest(x1v, x1_high_max)
    th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))

    dens_all = []
    mom1_all = []
    mom2_all = []
    mom3_all = []
    temp_all = []
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(kwargs['problem_id'] + ".prim." + str_t + ".athdf",
                                        quantities=['press'])
        data_cons = athena_read.athdf(kwargs['problem_id'] + ".cons." + str_t + ".athdf",
                                        quantities=['dens','mom1','mom2','mom3'])

        #unpack data
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        press = data_prim['press']
        temp = press/dens

        dens = dens[:, th_l:th_u, :]
        mom1 = mom1[:, th_l:th_u, :]
        mom2 = mom2[:, th_l:th_u, :]
        mom3 = mom3[:, th_l:th_u, :]
        temp = temp[:, th_l:th_u, :]

        dens_profile = np.average(dens, axis=(0,1))
        mom1_profile = np.average(mom1, axis=(0,1))
        mom2_profile = np.average(mom2, axis=(0,1))
        mom3_profile = np.average(mom3, axis=(0,1))
        temp_profile = np.average(temp, axis=(0,1))

        dens_all.append(dens_profile)
        mom1_all.append(mom1_profile)
        mom2_all.append(mom2_profile)
        mom3_all.append(mom3_profile)
        temp_all.append(temp_profile)

    comm.barrier()
    if rank == 0:
        dens_av = np.mean(dens_all, axis=0)
        mom1_av = np.mean(mom1_all, axis=0)
        mom2_av = np.mean(mom2_all, axis=0)
        mom3_av = np.mean(mom3_all, axis=0)
        temp_av = np.mean(temp_all, axis=0)

        np.save(kwargs['output'] + 'dens_profile.npy', dens_av)
        np.save(kwargs['output'] + 'mom1_profile.npy', mom1_av)
        np.save(kwargs['output'] + 'mom2_profile.npy', mom2_av)
        np.save(kwargs['output'] + 'mom3_profile.npy', mom3_av)
        np.save(kwargs['output'] + 'temp_profile.npy', temp_av)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, including path')
    parser.add_argument('input',
                        help='location of athinput file, including path')
    parser.add_argument('output',
                        help='location of output folder, including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
