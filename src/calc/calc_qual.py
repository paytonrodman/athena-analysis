#!/usr/bin/env python3
#
# calc_qual.py
#
# A program to calculate the quality factors and magnetic angle within some defined region
# of an Athena++ disk using MPI
#
# Usage: mpirun -n [nprocs] calc_qual.py [options]
#
# Python standard modules
import argparse
import sys
import os
import gc
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
    x1max = data_input['mesh']['x1max']
    if 'refinement3' in data_input:
        x1_high_max = data_input['refinement3']['x1max']
    elif 'refinement2' in data_input:
        x1_high_max = data_input['refinement2']['x1max']
    elif 'refinement1' in data_input:
        x1_high_max = data_input['refinement1']['x1max']
    else:
        x1_high_max = x1max

    # retrieve lists of scale height with time
    if rank==0:
        df = pd.read_csv(args.scale, delimiter='\t', usecols=['sim_time', 'scale_height'])
        scale_time_list = df['sim_time'].to_list()
        scale_height_list = df['scale_height'].to_list()
    else:
        scale_time_list = None
        scale_height_list = None
    scale_height_list = comm.bcast(scale_height_list, root=0)
    scale_time_list = comm.bcast(scale_time_list, root=0)

    if rank==0:
        if not args.update:
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["file_time", "sim_time", "orbit_time", "theta_B", "Q_theta", "Q_phi"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        gamma = 5./3.
        GM = 1.

        #unpack data
        data_cons = athena_read.athdf(args.problem_id + '.cons.' + str_t + '.athdf',
                                        quantities=['dens','mom1','mom2','mom3',
                                                    'Bcc1','Bcc2','Bcc3'])
        data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list,data_cons['Time'])
        scale_height = scale_height_list[scale_index]

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        dens = data_cons['dens']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        # create 3D meshes
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        dphi,dtheta,_ = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

        # restrict range of meshes
        R = r*np.sin(theta)
        R = R[:, th_l:th_u, :r_u]
        r = r[:, th_l:th_u, :r_u]
        phi = phi[:, th_l:th_u, :r_u]
        dphi = dphi[:, th_l:th_u, :r_u]
        dtheta = dtheta[:, th_l:th_u, :r_u]

        #del dx1f, dx2f, dx3f
        #del x2v, x3v
        #del x1f, x2f, x3f
        #gc.collect()

        #del data_prim
        #gc.collect()

        # define bounds of region to average over
        r_u = AAT.find_nearest(x1v, x1_high_max)
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

        # restrict ranges
        dens = dens[:, th_l:th_u, :r_u]
        Bcc1 = Bcc1[:, th_l:th_u, :r_u]
        Bcc2 = Bcc2[:, th_l:th_u, :r_u]
        Bcc3 = Bcc3[:, th_l:th_u, :r_u]
        press = press[:, th_l:th_u, :r_u]

        # calculate Keplerian orbital velocity
        Omega_kep = np.sqrt(GM/(x1v[:r_u]**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))

        # magnetic tilt angle in degrees
        tB = (-np.arctan(Bcc1/Bcc3)) * (180./np.pi)
        tB_av = np.average(tB)

        # quality factors in theta and phi
        w = dens + (gamma/(gamma - 1.))*press
        B2 = Bcc1**2. + Bcc2**2. + Bcc3**2.
        vA_theta = Bcc2/(np.sqrt(w+B2)) #Alfven velocity of theta component of B
        vA_phi = Bcc3/(np.sqrt(w+B2)) #Alfven velocity of phi component of B

        lambda_MRI_theta = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_theta)/Omega_kep
        lambda_MRI_phi = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_phi)/Omega_kep

        Q_theta = lambda_MRI_theta/(r*dtheta)
        Q_phi = lambda_MRI_phi/(R*dphi)

        Q_theta = np.array(Q_theta.flatten())
        Q_phi = np.array(Q_phi.flatten())

        Qt_av = np.mean(Q_theta)
        Qp_av = np.mean(Q_phi)

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [t,sim_t,orbit_t,tB_av,Qt_av,Qp_av]
            writer.writerow(row)
        # cleanup
        #del w, B2, vA_theta, vA_phi, lambda_MRI_theta, lambda_MRI_phi
        #del R, r, phi, dphi, dtheta
        #del Q_theta, Q_phi
        #gc.collect()
        #print("data written for time ", sim_t)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        help='location of athinput file, possibly including path')
    parser.add_argument('scale',
                        help='location of scale height file, possibly including path')
    parser.add_argument('output',
                        help='name of output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    args = parser.parse_args()

    main(**vars(args))
