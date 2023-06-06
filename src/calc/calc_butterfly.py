#!/usr/bin/env python3
#
# calc_butterfly.py
#
# A program to calculate the azimuthally-averaged field of an Athena++ disk using MPI,
# for use in butterfly plots
#
# Usage: mpirun -n [nprocs] calc_butterfly.py [options]
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
import csv

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
    file_times = AAT.add_time_to_list(args.update, args.output)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    # define radius where data is collected
    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x1v'])
    x1v_init = data_init['x1v']
    if args.r is not None:
        r_id = AAT.find_nearest(x1v_init, args.r)
    else:
        r_id = AAT.find_nearest(x1v_init, 25.) # approx. middle of high res region

    if rank==0:
        if not args.update: # create output file with header
            with open(kargs.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['sim_time', 'orbit_time', 'Bcc1', 'Bcc2', 'Bcc3'])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['Bcc1','Bcc2','Bcc3'])

        #unpack data
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        Bcc1_theta = np.average(Bcc1[:,:,r_id],axis=0).tolist()
        Bcc2_theta = np.average(Bcc2[:,:,r_id],axis=0).tolist()
        Bcc3_theta = np.average(Bcc3[:,:,r_id],axis=0).tolist()

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([sim_t,orbit_t,Bcc1_theta,Bcc2_theta,Bcc3_theta])


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action='store_true',
                        help='append new results to an existing data file')
    parser.add_argument('--r',
                        type=float,
                        default=None,
                        help=('value of r where averages are calculated (default: 25rg)'))
    args = parser.parse_args()

    main(**vars(args))
