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

    data_init = athena_read.athdf(args.problem_id + '.cons.00000.athdf')
    x1v_init = data_init['x1v'] # r
    x2v_init = data_init['x2v'] # theta

    if args.r_lower is not None:
        if not x1min <= args.r_lower < x1max:
            sys.exit('Error: Lower r value must be between %d and %d' % x1min,x1max)
        rl = AAT.find_nearest(x1v_init, args.r_lower)
    else:
        rl = AAT.find_nearest(x1v_init, x1_high_min)
    if args.r_upper is not None:
        if not x1min <= args.r_upper < x1max:
            sys.exit('Error: Upper r value must be between %d and %d' % x1min,x1max)
        ru = AAT.find_nearest(x1v_init, args.r_upper)
    else:
        ru = AAT.find_nearest(x1v_init, x1_high_max)
    if args.theta_lower is not None:
        if not x2min <= args.theta_lower < x2max:
            sys.exit('Error: Lower theta value must be between %d and %d' % x2min,x2max)
        tl = AAT.find_nearest(x2v_init, args.theta_lower)
    else:
        tl = AAT.find_nearest(x2v_init, x2_high_min)
    if args.theta_upper is not None:
        if not x2min <= args.theta_upper < x2max:
            sys.exit('Error: Upper theta value must be between %d and %d' % x2min,x2max)
        tu = AAT.find_nearest(x2v_init, args.theta_upper)
    else:
        tu = AAT.find_nearest(x2v_init, x2_high_max)

    if rl==ru:
        ru += 1
    if tl==tu:
        tu += 1

    del x2v_init, x1v_init

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
                                        quantities=['x1v','x2v','x3v','x1f','x2f','x3f',
                                                    'dens','mom1','mom2','mom3',
                                                    'Bcc1','Bcc2','Bcc3'])
        data_prim = athena_read.athdf(args.problem_id + '.prim.' + str_t + '.athdf',
                                        quantities=['press'])

        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        dens = data_cons['dens'][:, tl:tu, rl:ru]
        Bcc1 = data_cons['Bcc1'][:, tl:tu, rl:ru]
        Bcc2 = data_cons['Bcc2'][:, tl:tu, rl:ru]
        Bcc3 = data_cons['Bcc3'][:, tl:tu, rl:ru]
        press = data_prim['press'][:, tl:tu, rl:ru]
        sim_t = data_cons['Time']

        orbit_t = AAT.calculate_orbit_time(sim_t)
        del data_cons, data_prim
        gc.collect()

        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        phi,theta,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        dphi,dtheta,_ = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

        del dx1f, dx2f, dx3f
        del x2v, x3v
        del x1f, x2f, x3f
        gc.collect()

        Omega_kep = np.sqrt(GM/(x1v[rl:ru]**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))

        tB = (-np.arctan(Bcc1/Bcc3)) * (180./np.pi) #magnetic tilt angle in degrees
        tB_av = np.average(tB)

        w = dens + (gamma/(gamma - 1.))*press
        B2 = Bcc1**2. + Bcc2**2. + Bcc3**2.
        vA_theta = Bcc2/(np.sqrt(w+B2)) #Alfven velocity of theta component of B
        vA_phi = Bcc3/(np.sqrt(w+B2)) #Alfven velocity of phi component of B
        lambda_MRI_theta = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_theta)/Omega_kep
        lambda_MRI_phi = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_phi)/Omega_kep

        R = r*np.sin(theta)
        R = R[:, tl:tu, rl:ru]
        r = r[:, tl:tu, rl:ru]
        phi = phi[:, tl:tu, rl:ru]
        dphi = dphi[:, tl:tu, rl:ru]
        dtheta = dtheta[:, tl:tu, rl:ru]

        Q_theta = lambda_MRI_theta/(r*dtheta)
        Q_phi = lambda_MRI_phi/(R*dphi)
        Q_theta = np.array(Q_theta.flatten())
        Q_phi = np.array(Q_phi.flatten())

        Qt_av = np.mean(Q_theta)
        Qp_av = np.mean(Q_phi)

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [t,sim_t,orbit_t,tB_av,Qt_av,Qp_av]
            writer.writerow(row)
        # cleanup
        del w, B2, vA_theta, vA_phi, lambda_MRI_theta, lambda_MRI_phi
        del R, r, phi, dphi, dtheta
        del Q_theta, Q_phi
        gc.collect()
        print("data written for time ", sim_t)


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
                        default=None,
                        help='value of lower theta bound of region being analysed, must be between x2min and x2max (default=0.982)')
    parser.add_argument('-tu', '--theta_upper',
                        type=float,
                        default=None,
                        help='value of upper theta bound of region being analysed, must be between x2min and x2max (default=2.159)')
    args = parser.parse_args()

    main(**vars(args))
