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

    os.chdir(args.data)

    file_times = AAT.add_time_to_list(args.update, args.output)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(args.input)
    scale_height = data_input['problem']['h_r']
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max'] # bounds of high resolution region
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max'] # bounds of high resolution region
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max'] # bounds of high resolution region
    else:
        x1_high_max = data_input['mesh']['x1max']

    data_init = athena_read.athdf(kargs.problem_id + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r_u = AAT.find_nearest(x1v, x1_high_max)
    th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

    if rank==0:
        if not args.update:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "plasma_beta"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.problem_id + ".prim." + str_t + ".athdf",
                                        quantities=['press'])
        data_cons = athena_read.athdf(args.problem_id + ".cons." + str_t + ".athdf",
                                        quantities=['x1f','x2f','x3f','dens','Bcc1','Bcc2','Bcc3'])

        #unpack data
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        r,theta,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')
        dphi = dphi[:, th_l:th_u, :r_u]
        dtheta = dtheta[:, th_l:th_u, :r_u]
        dr = dr[:, th_l:th_u, :r_u]
        r = r[:, th_l:th_u, :r_u]
        theta = theta[:, th_l:th_u, :r_u]
        pressure = press[:, th_l:th_u, :r_u]
        density = dens[:, th_l:th_u, :r_u]

        # Density-weighted mean gas pressure
        sum_p = 0.
        numWeight_p = 0.
        sum_b = 0.
        numWeight_b = 0.

        volume = (r**2.)*np.sin(theta)*dr*dtheta*dphi

        # Find volume centred total magnetic field
        bcc_all = np.square(Bcc1[:, th_l:th_u, :r_u]) + np.square(Bcc2[:, th_l:th_u, :r_u]) + np.square(Bcc3[:, th_l:th_u, :r_u])
        # Density- and volume weighted pressure/magnetic field
        numWeight_p = np.sum(pressure*density*volume) #value * weight
        sum_p       = np.sum(density*volume) # weight
        numWeight_b = np.sum(bcc_all*density*volume)
        sum_b       = np.sum(density*volume)

        pres_av = numWeight_p/sum_p
        bcc_av = numWeight_b/sum_b
        if bcc_av>0:
            current_beta = 2. * pres_av / bcc_av
        else:
            current_beta = np.nan

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,current_beta]
            writer.writerow(row)

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate plasma beta from raw simulation data.')
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
