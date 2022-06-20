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
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import scipy.stats as st
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

    os.chdir(kwargs['data'])

    data_input = athena_read.athinput(kwargs['input'])
    x1min = data_input['mesh']['x1min'] # bounds of simulation
    x1max = data_input['mesh']['x1max']
    x2min = data_input['mesh']['x2min']
    x2max = data_input['mesh']['x2max']
    if 'refinement3' in data_input:
        x1_high_min = data_input['refinement3']['x1min'] # bounds of high resolution region
        x1_high_max = data_input['refinement3']['x1max']
        x2_high_min = data_input['refinement3']['x2min']
        x2_high_max = data_input['refinement3']['x2max']
    elif 'refinement2' in data_input:
        x1_high_min = data_input['refinement2']['x1min'] # bounds of high resolution region
        x1_high_max = data_input['refinement2']['x1max']
        x2_high_min = data_input['refinement2']['x2min']
        x2_high_max = data_input['refinement2']['x2max']
    elif 'refinement1' in data_input:
        x1_high_min = data_input['refinement1']['x1min'] # bounds of high resolution region
        x1_high_max = data_input['refinement1']['x1max']
        x2_high_min = data_input['refinement1']['x2min']
        x2_high_max = data_input['refinement1']['x2max']
    else:
        x1_high_min = x1min
        x1_high_max = x1max
        x2_high_min = x2min
        x2_high_max = x2max

    data_init = athena_read.athdf(kwargs['problem_id'] + '.cons.00000.athdf',
                                    quantities=['x1v','x2v'])
    x1v_init = data_init['x1v'] # r
    x2v_init = data_init['x2v'] # theta

    if kwargs['theta_lower'] is not None:
        if not x2min <= kwargs['theta_lower'] < x2max:
            sys.exit('Error: Lower theta value must be between %d and %d' % x2min,x2max)
        tl = AAT.find_nearest(x2v_init, kwargs['theta_lower'])
    else:
        tl = AAT.find_nearest(x2v_init, x2_high_min)
    if kwargs['theta_upper'] is not None:
        if not x2min <= kwargs['theta_upper'] < x2max:
            sys.exit('Error: Upper theta value must be between %d and %d' % x2min,x2max)
        tu = AAT.find_nearest(x2v_init, kwargs['theta_upper'])
    else:
        tu = AAT.find_nearest(x2v_init, x2_high_max)

    file_times = AAT.add_time_to_list(kwargs['update'], kwargs['Qtheta_output'])
    if kwargs['problem_id']=='high_res':
        file_times = file_times[file_times>2.5e4]
    elif kwargs['problem_id']=='high_beta' or kwargs['problem_id']=='super_res':
        file_times = file_times[file_times>1e4]
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    Qphi_all = []
    Qtheta_all = []
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        gamma = 5./3.
        GM = 1.

        #unpack data
        data_cons = athena_read.athdf(kwargs['problem_id'] + '.cons.' + str_t + '.athdf',
                                        quantities=['x1v','x2v','x3v','x1f','x2f','x3f',
                                                    'dens','mom1','mom2','mom3',
                                                    'Bcc1','Bcc2','Bcc3'])
        data_prim = athena_read.athdf(kwargs['problem_id'] + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])

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

        Omega_kep = np.sqrt(GM/(x1v**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))

        tB = (-np.arctan(Bcc1/Bcc3)) * (180./np.pi) #magnetic tilt angle in degrees
        tB_av = np.average(tB)

        w = dens + (gamma/(gamma - 1.))*press
        B2 = Bcc1**2. + Bcc2**2. + Bcc3**2.
        vA_theta = Bcc2/(np.sqrt(w+B2)) #Alfven velocity of theta component of B
        vA_phi = Bcc3/(np.sqrt(w+B2)) #Alfven velocity of phi component of B
        lambda_MRI_theta = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_theta)/Omega_kep
        lambda_MRI_phi = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_phi)/Omega_kep

        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        dphi,dtheta,_ = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

        R = r*np.sin(theta)

        #Q_theta = lambda_MRI_theta/np.sqrt(r*dtheta)
        Q_theta = lambda_MRI_theta/(R*dtheta)
        Q_theta = np.mean(Q_theta, axis=0) #average azimuthally
        #Q_phi = lambda_MRI_phi/np.sqrt(r*np.abs(np.sin(phi))*dphi)
        Q_phi = lambda_MRI_phi/(R*dphi)
        Q_phi = np.mean(Q_phi[:, tl:tu, :], axis=1) #average within high res theta range

        Qtheta_all.append(Q_theta)
        Qphi_all.append(Q_phi)

    comm.barrier()
    if rank == 0:
        Qtheta_av = np.mean(Qtheta_all, axis=0)
        Qphi_av = np.mean(Qphi_all, axis=0)

        np.save(kwargs['Qtheta_output'],Qtheta_av)
        np.save(kwargs['Qphi_output'],Qphi_av)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('problem_id',
                        help='root name for data files, e.g. high_res')
    parser.add_argument('data',
                        help='location of data folder, possibly including path')
    parser.add_argument('input',
                        help='location of athinput file, possibly including path')
    parser.add_argument('Qtheta_output',
                        help='name of Qtheta output to be (over)written, possibly including path')
    parser.add_argument('Qphi_output',
                        help='name of Qphi output to be (over)written, possibly including path')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='append new results to an existing data file')
    parser.add_argument('-rl', '--r_lower',
                        type=float,
                        default=None,
                        help='value of lower r bound of region being analysed, must be between x1min and x1max (default=5)')
    parser.add_argument('-ru', '--r_upper',
                        type=float,
                        default=None,
                        help='value of upper r bound of region being analysed, must be between x1min and x1max (default=100)')
    parser.add_argument('-tl', '--theta_lower',
                        type=float,
                        default=0.982,
                        help='value of lower theta bound of region being analysed, must be between x2min and x2max (default=0.982)')
    parser.add_argument('-tu', '--theta_upper',
                        type=float,
                        default=2.159,
                        help='value of upper theta bound of region being analysed, must be between x2min and x2max (default=2.159)')
    args = parser.parse_args()

    main(**vars(args))
