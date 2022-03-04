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
    runfile_dir = prob_dir + 'runfiles/'
    filename_output = 'beta_with_time.csv'
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
            if float(current_time[0]) not in file_times:
                file_times = np.append(file_times, float(current_time[0]))
    if len(file_times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    # distribute files to cores
    files_per_process = len(file_times) // size
    remainder = len(file_times) % size
    if rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = rank * (files_per_process + 1)
        stop = start + files_per_process + 1
    else:
        start = rank * files_per_process + remainder
        stop = start + files_per_process

    local_times = file_times[start:stop] # get the times to be analyzed by each rank

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + args.prob_id)
    scale_height = data_input['problem']['h_r']
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max'] # bounds of high resolution region
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max'] # bounds of high resolution region
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max'] # bounds of high resolution region
    else:
        x1_high_max = data_input['mesh']['x1max']

    data_init = athena_read.athdf(args.prob_id + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r_u = AAT.find_nearest(x1v, x1_high_max)
    th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

    del data_input
    del data_init

    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "plasma_beta"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(args.prob_id + ".prim." + str_t + ".athdf", quantities=['press'])
        data_cons = athena_read.athdf(args.prob_id + ".cons." + str_t + ".athdf", quantities=['x1f','x2f','x3f','dens','Bcc1','Bcc2','Bcc3'])

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

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + filename_output, 'a', newline='') as f:
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
