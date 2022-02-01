#!/usr/bin/env python3
#
# calc_qual.py
#
# A program to calculate the quality factors and magnetic angle within some defined region
# of an Athena++ disk using MPI
#
# To run:
# mpirun -n [n] python calc_qual.py [options]
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
import scipy.stats as st
import argparse
from math import sqrt
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

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
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

    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x1v','x2v'])
    x1v_init = data_init['x1v'] # r
    x2v_init = data_init['x2v'] # theta

    if kwargs['r_lower'] is not None:
        if not x1min <= kwargs['r_lower'] < x1max:
            sys.exit('Error: Lower r value must be between %d and %d' % x1min,x1max)
        rl = AAT.find_nearest(x1v_init, kwargs['r_lower'])
    else:
        rl = AAT.find_nearest(x1v_init, x1_high_min)
    if kwargs['r_upper'] is not None:
        if not x1min <= kwargs['r_upper'] < x1max:
            sys.exit('Error: Upper r value must be between %d and %d' % x1min,x1max)
        ru = AAT.find_nearest(x1v_init, kwargs['r_upper'])
    else:
        ru = AAT.find_nearest(x1v_init, x1_high_max)
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

    if rl==ru:
        ru += 1
    if tl==tu:
        tu += 1

    filename_output = 'qual_with_time_' + str(rl) + '_' + str(ru) + '_' + str(tl) + '_' + str(tu) + '.csv'
    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + filename_output, 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time = np.append(csv_time, float(row[0]))

    files = glob.glob('./high_res.cons.*.athdf')
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
    if rank < remainder:  # processes with rank < remainder analyze one extra file
        start = rank * (count + 1)  # index of first file to analyze
        stop = start + count + 1  # index of last file to analyze
    else:
        start = rank * count + remainder
        stop = start + count
    local_times = times[start:stop] # get the times to be analyzed by each rank

    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time", "theta_B", "Q_theta", "Q_phi"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        #constants
        gamma = 5./3.
        GM = 1.

        #unpack data
        data_prim = athena_read.athdf(problem + '.prim.' + str_t + '.athdf', quantities=['press'])
        press = data_prim['press']

        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['x1v','x2v','x3v','x1f','x2f','x3f','dens','mom1','mom2','mom3','Bcc1','Bcc2','Bcc3'])
        x1v = data_cons['x1v'] # r
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        Omega_kep = np.empty_like(dens)

        # Calculations
        dx1f,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        del x1f,x2f,x3f

        for ii in range(0,256):
            Omega_kep[ii, :, :] = np.sqrt(GM/(x1v**3.)) #Keplerian angular velocity in midplane

        Bcc1 = Bcc1[:, tl:tu, rl:ru]
        Bcc2 = Bcc2[:, tl:tu, rl:ru]
        Bcc3 = Bcc3[:, tl:tu, rl:ru]
        dens = dens[:, tl:tu, rl:ru]
        mom1 = mom1[:, tl:tu, rl:ru]
        mom2 = mom2[:, tl:tu, rl:ru]
        mom3 = mom3[:, tl:tu, rl:ru]
        press = press[:, tl:tu, rl:ru]
        Omega_kep = Omega_kep[:, tl:tu, rl:ru]

        tB = (-np.arctan(Bcc1/Bcc3)) * (180./np.pi) #degrees
        tB_av = np.average(tB)

        w = dens + (gamma/(gamma - 1.))*press
        B2 = Bcc1**2. + Bcc2**2. + Bcc3**2.
        vA_theta = Bcc2/(np.sqrt(w+B2)) #Alfven velocity of theta component of B
        vA_phi = Bcc3/(np.sqrt(w+B2)) #Alfven velocity of phi component of B
        lambda_MRI_theta = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_theta)/Omega_kep
        lambda_MRI_phi = 2.*np.pi*np.sqrt(16./15.)*np.abs(vA_phi)/Omega_kep

        phi,_,r = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        dphi,dtheta,_ = np.meshgrid(dx3f,dx2f,dx1f, sparse=False, indexing='ij')

        r = r[:, tl:tu, rl:ru]
        phi = phi[:, tl:tu, rl:ru]
        dphi = dphi[:, tl:tu, rl:ru]
        dtheta = dtheta[:, tl:tu, rl:ru]

        Q_theta = lambda_MRI_theta/np.sqrt(r*dtheta)
        Q_phi = lambda_MRI_phi/np.sqrt(r*np.abs(np.sin(phi))*dphi)
        Q_theta = np.array(Q_theta.flatten())
        Q_phi = np.array(Q_phi.flatten())

        Qt_l,Qt_h = st.t.interval(0.95, len(Q_theta)-1, loc=np.mean(Q_theta), scale=st.sem(Q_theta))
        Qt_av = np.mean(Q_theta)
        Qp_l,Qp_h = st.t.interval(0.95, len(Q_phi)-1, loc=np.mean(Q_phi), scale=st.sem(Q_phi))
        Qp_av = np.mean(Q_phi)

        r_ISCO = 6 # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        Qt_all = [Qt_l,Qt_av,Qt_h]
        Qp_all = [Qp_l,Qp_av,Qp_h]

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,tB_av,Qt_all,Qp_all]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
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
