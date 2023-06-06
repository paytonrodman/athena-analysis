#!/usr/bin/env python3
#
# calc_pressure_profiles.py
#
# A program to calculate the radial pressure profile of a disk using MPI.
#
# Usage: mpirun -n [nprocs] calc_pressure_profiles.py [options]
#
# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np
from mpi4py import MPI

# Athena++ modules (require sys.path.append above)
import athena_read
import AAT

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(args.data)

    # make list of files/times to analyse, distribute to cores
    file_times = AAT.add_time_to_list(False, None) # retrieve all data file names
    file_times.sort()
    _,_,t_min = AAT.problem_dictionary(args.problem_id, False) # get minimum required time
    file_times_restricted = []
    if rank==0:
        for f in file_times:
            str_f = str(int(f)).zfill(5)
            # read in some small slice of the file to check the time
            data_check = athena_read.athdf(args.problem_id + '.cons.' + str_f + '.athdf',
                                           x1_min=5,x1_max=6,x2_min=0,x2_max=0.1,x3_min=0,x3_max=0.1)
            sim_t = data_check['Time']
            if sim_t >= t_min:
                file_times_restricted.append(f)
    else:
        file_times_restricted = None
    comm.barrier() # wait for master node to be done before moving on
    file_times_restricted = comm.bcast(file_times_restricted, root=0) # broadcast list to all nodes

    if not file_times_restricted: # if list is empty
        sys.exit('No file times meet requirements. Exiting.')

    local_times = AAT.distribute_files_to_cores(file_times_restricted, size, rank)

    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    r = AAT.find_nearest(x1v, 15.)

    gpres_all = []
    mpres_all = []
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['Bcc1','Bcc2','Bcc3'])

        #unpack data
        gpres = data_prim['press']
        mpres = (data_cons['Bcc1']**2. + data_cons['Bcc2']**2. + data_cons['Bcc3']**2.)/2.0

        gpres = gpres[:, :, r]
        mpres = mpres[:, :, r]

        gpres_profile = np.average(gpres, axis=(0))
        mpres_profile = np.average(mpres, axis=(0))

        gpres_all.append(gpres_profile)
        mpres_all.append(mpres_profile)

    comm.barrier()
    if rank == 0:
        gpres_av = np.mean(gpres_all, axis=0)
        mpres_av = np.mean(mpres_all, axis=0)

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
                        action='store_true',
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
