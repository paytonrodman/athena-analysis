#!/usr/bin/env python3
#
# calc_mass.py
#
# A program to calculate the mass accretion rate of an Athena++ disk using MPI.
#
# To run:
# mpirun -n [n] python calc_mass.py [options]
# for [n] cores.
import numpy as np
import os
import sys
sys.path.append('/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
from math import sqrt
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
    filename_output = 'mass_with_time.csv'
    os.chdir(data_dir)

    # check if data file already exists
    csv_times = np.empty(0)
    if args.update:
        with open(prob_dir + filename_output, 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_times = np.append(csv_times, float(row[0]))

    # compile a list of unique times associated with data files
    files = glob.glob('./' + args.prob_id + '.cons.*.athdf')
    file_times = np.empty(0)
    for f in files:
        current_time = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(current_time[0]) not in file_times and float(current_time[0]) not in csv_times:
                file_times = np.append(file_times, float(current_time[0]))
        else:
            if float(current_time[0]) not in times:
                file_times = np.append(file_times, float(current_time[0]))
    if len(file_times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    init_data = athena_read.athdf(args.prob_id + '.cons.00000.athdf', quantities=['x1v'])
    x1v_init = init_data['x1v'] # r
    r_val = [6.,25.,50.,75.,100.]
    r_id = []
    for r in r_val:
        r_id_i = AAT.find_nearest(x1v_init, r)
        r_id.append(r_id_i)

    # distribute files to cores
    files_per_process = len(times) // size
    remainder = len(times) % size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (files_per_process + 1)
        stop = start + files_per_process + 1
    else:
        start = rank * files_per_process + remainder
        stop = start + files_per_process
        
    local_times = times[start:stop] # get the times to be analyzed by each rank

    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "mass_flux"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.prob_id + '.cons.' + str_t + '.athdf', quantities=['x2v','x3v','x1f','x2f','x3f','dens','mom1','mom2','mom3'])

        #unpack data
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        # Calculations
        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        v1,_,_ = AAT.calculate_velocity(mom1,mom2,mom3,dens)

        mf_local = []
        for r_id_i in r_id:
            mf_i = []
            for j in range(len(x2v)):
                for k in range(len(x3v)):
                    dOmega = np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                    mf_i.append(-dens[k,j,r_id_i] * v1[k,j,r_id_i] * (x1f[r_id_i])**2. * dOmega)
            mf_local.append(np.sum(mf_i))

        r_ISCO = 6 # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,mf_local]
            writer.writerow(row)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate instantaneous mass flux across inner radial boundary')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
