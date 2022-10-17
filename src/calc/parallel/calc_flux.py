#!/usr/bin/env python3
#
# calc_flux.py
#
# A program to calculate the magnetic flux threading the ISCO of an Athena++ disk using MPI
#
# Usage: mpirun -n [nprocs] calc_flux.py [options]
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

    os.chdir(args.data)

    file_times = AAT.add_time_to_list(args.update, args.output)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(args.input)
    scale_height = data_input['problem']['h_r']

    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf', quantities=['x1v', 'x2v'])
    x1v_init = data_init['x1v']
    x2v_init = data_init['x2v']
    r_id = AAT.find_nearest(x1v_init, 6.)
    th_id = AAT.find_nearest(x2v_init, np.pi/2.)

    th_l = AAT.find_nearest(x2v_init, np.pi/2. - (2.*scale_height))
    th_u = AAT.find_nearest(x2v_init, np.pi/2. + (2.*scale_height))

    if rank==0:
        if not args.update:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["file_time", "sim_time", "orbit_time",
                                    "mag_flux_u", "mag_flux_l",
                                    "mag_flux_u_abs", "mag_flux_l_abs",
                                    "mag_flux_u_disk", "mag_flux_l_disk",
                                    "mag_flux_u_abs_disk", "mag_flux_l_abs_disk"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['x2v','x3v','x1f','x2f','x3f','dens','mom1','Bcc1'])

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
        mf_l_d = []
        mf_u_d = []
        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[r_id]**2.) * np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf_u.append(Bcc1[k,j,r_id]*dS)
                if j<th_l:
                    mf_u_d.append(Bcc1[k,j,r_id]*dS)
        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[r_id]**2.) * np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf_l.append(Bcc1[k,j,r_id]*dS)
                if j>th_u:
                    mf_l_d.append(Bcc1[k,j,r_id]*dS)


        mag_flux_u = np.sqrt(4.*np.pi) * np.sum(mf_u)
        mag_flux_u_abs = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_u))
        mag_flux_l = np.sqrt(4.*np.pi) * np.sum(mf_l)
        mag_flux_l_abs = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_l))

        mag_flux_u_disk = np.sqrt(4.*np.pi) * np.sum(mf_u_d)
        mag_flux_u_abs_disk = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_u_d))
        mag_flux_l_disk = np.sqrt(4.*np.pi) * np.sum(mf_l_d)
        mag_flux_l_abs_disk = np.sqrt(4.*np.pi) * np.sum(np.abs(mf_l_d))

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [int(t), sim_t, orbit_t,
                    mag_flux_u, mag_flux_l,
                    mag_flux_u_abs, mag_flux_l_abs,
                    mag_flux_u_disk, mag_flux_l_disk,
                    mag_flux_u_abs_disk, mag_flux_l_abs_disk]
            writer.writerow(row)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        help='location of athinput file, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
