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
    if args.problem_id=='high_res':
        file_times = file_times[file_times>10000] # t > 5e4
    elif args.problem_id=='high_beta' or args.problem_id=='super_res':
        file_times = file_times[file_times>4000] # t > 2e4
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(args.input)
    scale_height = data_input['problem']['h_r']

    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r = AAT.find_nearest(x1v, 15.)
    #th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))
    #th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))

    dens_all = []
    gpres_all = []
    mpres_all = []
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + ".prim." + str_t + ".athdf",
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + ".cons." + str_t + ".athdf",
                                        quantities=['dens','mom1','mom2','mom3','Bcc1','Bcc2','Bcc3'])

        #unpack data
        dens = data_cons['dens']
        gpres = data_prim['press']
        mpres = (data_cons['Bcc1']**2. + data_cons['Bcc2']**2. + data_cons['Bcc3']**2.)/2.0

        dens = dens[:, :, r]
        gpres = gpres[:, :, r]
        mpres = mpres[:, :, r]

        dens_profile = np.average(dens, axis=(0))
        gpres_profile = np.average(gpres, axis=(0))
        mpres_profile = np.average(mpres, axis=(0))

        dens_all.append(dens_profile)
        gpres_all.append(gpres_profile)
        mpres_all.append(mpres_profile)

    comm.barrier()
    if rank == 0:
        dens_av = np.mean(dens_all, axis=0)
        gpres_av = np.mean(gpres_all, axis=0)
        mpres_av = np.mean(mpres_all, axis=0)

        np.save(args.output + 'dens_profile_th.npy', dens_av)
        np.save(args.output + 'gpres_profile_th.npy', gpres_av)
        np.save(args.output + 'mpres_profile_th.npy', mpres_av)

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
