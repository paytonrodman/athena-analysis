#!/usr/bin/env python3
#
# calc_beta.py
#
# A program to calculate the plasma beta of an Athena++ disk using MPI.
#
# To run:
# mpirun -n [n] python calc_beta.py [options]
# for [n] cores.
#
import numpy as np
import os
import sys
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
from mpi4py import MPI

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    problem  = args.prob_id
    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
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

    files = glob.glob('./*.athdf')
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

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    scale_height = data_input['problem']['h_r']
    mass = data_input['problem']['mass']
    x1min = data_input['mesh']['x1min']
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
        data_prim = athena_read.athdf(problem + ".prim." + str_t + ".athdf", quantities=['press'])
        data_cons = athena_read.athdf(problem + ".cons." + str_t + ".athdf", quantities=['x1f','x2f','x3f','dens','Bcc1','Bcc2','Bcc3'])

        #unpack data
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        r,theta,phi = np.meshgrid(x3f,x2f,x1f, sparse=False, indexing='ij')
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        dphi,dtheta,dr = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

        dphi = dphi[:r_u,th_l:th_u,:]
        dtheta = dtheta[:r_u,th_l:th_u,:]
        dr = dr[:r_u,th_l:th_u,:]
        r = r[:r_u,th_l:th_u,:-1] # one extra face cell
        theta = theta[:r_u,th_l:th_u,:-1]
        pressure = press[:r_u,th_l:th_u,:]
        density = dens[:r_u,th_l:th_u,:]

        # Density-weighted mean gas pressure
        sum_p = 0.
        numWeight_p = 0.
        sum_b = 0.
        numWeight_b = 0.

        volume = (r**2.)*np.sin(theta)*dr*dtheta*dphi
        # Find volume centred total magnetic field
        bcc_all = np.sqrt(np.square(Bcc1[:r_u,th_l:th_u,:]) +
                          np.square(Bcc2[:r_u,th_l:th_u,:]) +
                          np.square(Bcc3[:r_u,th_l:th_u,:]))

        numWeight_p = np.sum(pressure*density*volume)
        sum_p       = np.sum(density*volume)
        numWeight_b = np.sum(bcc_all*density*volume)
        sum_b       = np.sum(density*volume)

        pres_av = numWeight_p/sum_p
        bcc_av = numWeight_b/sum_b
        if bcc_av>0:
            current_beta = 2. * pres_av / (bcc_av**2.)
        else:
            current_beta = np.nan

        v_Kep0 = np.sqrt(mass/x1min)
        Omega0 = v_Kep0/x1min
        T0 = 2.*np.pi/Omega0
        orbit_t = t/T0
        sim_t = float(t)

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
