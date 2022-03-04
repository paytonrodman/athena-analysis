#!/usr/bin/env python3
#
# calc_flux.py
#
# A program to calculate the magnetic flux threading the ISCO of an Athena++ disk using MPI
#
# To run:
# mpirun -n [n] python calc_flux.py [options]
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
    filename_output = 'flux_with_time.csv'
    os.chdir(data_dir)

    file_times = AAT.add_time_to_list(args.update, prob_dir, filename_output, args.prob_id)
    local_times = distribute_files_to_cores(file_times, size, rank)

    data_init = athena_read.athdf(args.prob_id + '.cons.00000.athdf', quantities=['x2v'])
    x2v_init = data_init['x2v']
    th_id = AAT.find_nearest(x2v_init, np.pi/2.)

    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "mag_flux_u", "mag_flux_l", "mag_flux_u_abs", "mag_flux_l_abs"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.prob_id + '.cons.' + str_t + '.athdf', quantities=['x2v','x3v','x1f','x2f','x3f','Bcc1'])

        #unpack data
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        Bcc1 = data_cons['Bcc1']

        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)

        mf_l = []
        mf_u = []
        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_u.append(Bcc1[k,j,0]*dS)
        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_l.append(Bcc1[k,j,0]*dS)

        mag_flux_u = np.sum(mf_u)
        mag_flux_u_abs = np.sum(np.abs(mf_u))
        mag_flux_l = np.sum(mf_l)
        mag_flux_l_abs = np.sum(np.abs(mf_l))

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,mag_flux_u,mag_flux_l,mag_flux_u_abs,mag_flux_l_abs]
            writer.writerow(row)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
