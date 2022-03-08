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
import csv
import argparse
import numpy as np
from mpi4py import MPI

def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + kwargs['prob_id'] + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    filename_output = 'beta_with_time.csv'
    os.chdir(data_dir)

    file_times = AAT.add_time_to_list(kwargs['update'], prob_dir, filename_output, kwargs['prob_id'])
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + kwargs['prob_id'])
    scale_height = data_input['problem']['h_r']
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max'] # bounds of high resolution region
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max'] # bounds of high resolution region
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max'] # bounds of high resolution region
    else:
        x1_high_max = data_input['mesh']['x1max']

    data_init = athena_read.athdf(kwargs['prob_id'] + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v = data_init['x1v']
    x2v = data_init['x2v']
    r_u = AAT.find_nearest(x1v, x1_high_max)
    th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

    if rank==0:
        if not kwargs['update']:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "plasma_beta"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_prim = athena_read.athdf(kwargs['prob_id'] + ".prim." + str_t + ".athdf", quantities=['press'])
        data_cons = athena_read.athdf(kwargs['prob_id'] + ".cons." + str_t + ".athdf", quantities=['x1f','x2f','x3f','dens','Bcc1','Bcc2','Bcc3'])

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
