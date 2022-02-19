#!/usr/bin/env python3
#
# calc_B.py
#
# A program to calculate the effective magnetic field derived from the magnetic flux at the inner
# simulation edge, and compare to the average field in the disk.
#
# Usage: python calc_B.py [options]
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

    # check if data file already exists
    csv_time = np.empty(0)
    if args.update:
        with open(prob_dir + 'B_strength_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time = np.append(csv_time, float(row[0]))

    files = glob.glob('./'+problem+'.cons.*.athdf')
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

    #local_times = [0.,5000.,10000.,15000.,20000.,25000.,30000.]

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    if 'refinement3' not in data_input:
        sys.exit('Simulation must have 3 levels of refinement in mesh. Exiting.')
    x1min = data_input['mesh']['x1min']
    x1max = data_input['refinement3']['x1max']
    scale_height = data_input['problem']['h_r']

    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x2v'])
    x2v = data_init['x2v']
    th_id = AAT.find_nearest(x2v, np.pi/2.)

    #jet_min_l = AAT.find_nearest(x2v, 0.0)
    jet_max_l = AAT.find_nearest(x2v, data_input['refinement1']['x2min'])
    upatmos_max_l = AAT.find_nearest(x2v, data_input['refinement2']['x2min'])
    loatmos_max_l = AAT.find_nearest(x2v, data_input['refinement3']['x2min'])
    disk_max_l = AAT.find_nearest(x2v, np.pi/2.)

    disk_max_u = AAT.find_nearest(x2v, data_input['refinement3']['x2max'])
    loatmos_max_u = AAT.find_nearest(x2v, data_input['refinement2']['x2max'])
    upatmos_max_u = AAT.find_nearest(x2v, data_input['refinement1']['x2max'])
    jet_max_u = AAT.find_nearest(x2v, np.pi)

    B_flux = []
    B_jet = []
    B_upatmos = []
    B_loatmos = []
    B_disk = []
    sim_time = []
    if rank==0:
        with open(prob_dir + 'B_strength_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["sim_time", "B_flux", "B_jet", "B_upatmos", "B_loatmos", "B_disk"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['x2v','x3v','x1f','x2f','x3f','Bcc1','Bcc2','Bcc3'])

        #unpack data
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        # Calculations
        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        B = np.abs(np.sqrt(Bcc1**2. + Bcc2**2. + Bcc3**2.))

        mf_l = []
        mf_l_abs = []
        mf_u = []
        mf_u_abs = []
        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_u.append(Bcc1[k,j,0]*dS)
                mf_u_abs.append(np.abs(Bcc1[k,j,0])*dS)
        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_l.append(Bcc1[k,j,0]*dS)
                mf_l_abs.append(np.abs(Bcc1[k,j,0])*dS)

        mag_flux_u = np.sum(mf_u)
        mag_flux_u_abs = np.sum(mf_u_abs)
        mag_flux_l = np.sum(mf_l)
        mag_flux_l_abs = np.sum(mf_l_abs)

        Bflux_l = mag_flux_l / (2.*np.pi*(x1min**2.))
        Bflux_u = mag_flux_u / (2.*np.pi*(x1min**2.))
        B_flux = [Bflux_l,Bflux_u]

        B_jet = [ np.average(B[:,:jet_max_l,:x1max]), np.average(B[:,upatmos_max_u+1:jet_max_u,:x1max]) ]
        B_upatmos = [ np.average(B[:,jet_max_l+1:upatmos_max_l,:x1max]), np.average(B[:,loatmos_max_u+1:upatmos_max_u,:x1max]) ]
        B_loatmos = [ np.average(B[:,upatmos_max_l+1:loatmos_max_l,:x1max]), np.average(B[:,disk_max_u+1:loatmos_max_u,:x1max]) ]
        B_disk = [ np.average(B[:,loatmos_max_l+1:disk_max_l,:x1max]), np.average(B[:,disk_max_l+1:disk_max_u,:x1max]) ]

        r_ISCO = 6 # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t = data_cons['Time']
        orbit_t = sim_t/T_period

        with open(prob_dir + 'B_strength_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,B_flux,B_jet,B_upatmos,B_loatmos,B_disk]
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
