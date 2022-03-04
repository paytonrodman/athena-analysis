#!/usr/bin/env python3
#
# calc_B.py
#
# A program to calculate the effective magnetic field derived from the magnetic flux at the inner
# simulation edge, and compare to the average field in the disk.
#
# Usage: python calc_B.py [options]
#
import os
import sys
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import csv
import argparse
from math import sqrt
import numpy as np
from mpi4py import MPI


def main(**kwargs):
    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # check user has passed valid --c variable
    if args.component not in ['r','theta','phi','all']:
        sys.exit('Must specify a valid B component')

    #root_dir = "/Users/paytonrodman/athena-sim/"
    root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + args.prob_id + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    filename_output = 'B_strength_with_time_' + args.component + '.csv'
    os.chdir(data_dir)

    file_times = AAT.add_time_to_list(args.update, prob_dir, filename_output, args.prob_id)
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + args.prob_id)
    if 'refinement3' not in data_input:
        sys.exit('Simulation must have 3 levels of refinement in mesh. Exiting.')
    x1min = data_input['mesh']['x1min']
    x1max = data_input['refinement3']['x1max']

    data_init = athena_read.athdf(args.prob_id + '.cons.00000.athdf', quantities=['x2v'])
    x2v = data_init['x2v']
    th_id = AAT.find_nearest(x2v, np.pi/2.)

    jet_max_l = AAT.find_nearest(x2v, data_input['refinement1']['x2min'])
    upatmos_max_l = AAT.find_nearest(x2v, data_input['refinement2']['x2min'])
    loatmos_max_l = AAT.find_nearest(x2v, data_input['refinement3']['x2min'])
    disk_max_l = AAT.find_nearest(x2v, np.pi/2.)

    disk_max_u = AAT.find_nearest(x2v, data_input['refinement3']['x2max'])
    loatmos_max_u = AAT.find_nearest(x2v, data_input['refinement2']['x2max'])
    upatmos_max_u = AAT.find_nearest(x2v, data_input['refinement1']['x2max'])
    jet_max_u = AAT.find_nearest(x2v, np.pi)


    if rank==0:
        if not args.update:
            with open(prob_dir + filename_output, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time",
                "abs_B_flux", "abs_B_jet", "abs_B_upatmos", "abs_B_loatmos", "abs_B_disk",
                "sign_B_flux", "sign_B_jet", "sign_B_upatmos", "sign_B_loatmos", "sign_B_disk"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(args.prob_id + '.cons.' + str_t + '.athdf',
                    quantities=['x2v','x3v','x1f','x2f','x3f','Bcc1','Bcc2','Bcc3'])

        #unpack data
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        x1f = data_cons['x1f']
        x2f = data_cons['x2f']
        x3f = data_cons['x3f']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']

        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)
        if args.component=='all':
            B = np.sqrt(Bcc1**2. + Bcc2**2. + Bcc3**2.)
        elif args.component=='r':
            B = Bcc1
        elif args.component=='theta':
            B = Bcc2
        elif args.component=='phi':
            B = Bcc3

        mf_l = []
        mf_u = []
        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_u.append(Bcc1[k,j,0]*dS)
        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_l.append(Bcc1[k,j,0]*dS)

        abs_B_flux_l = np.sum(np.abs(mf_l)) / (2.*np.pi*(x1min**2.))
        abs_B_flux_u = np.sum(np.abs(mf_u)) / (2.*np.pi*(x1min**2.))
        abs_B_flux = [abs_B_flux_l, abs_B_flux_u]

        abs_B_jet_l = np.average(abs(B[:, :jet_max_l, :x1max]))
        abs_B_jet_u = np.average(abs(B[:, upatmos_max_u+1:jet_max_u, :x1max]))
        abs_B_jet = [abs_B_jet_l, abs_B_jet_u]

        abs_B_upatmos_l = np.average(abs(B[:, jet_max_l+1:upatmos_max_l, :x1max]))
        abs_B_upatmos_u = np.average(abs(B[:, loatmos_max_u+1:upatmos_max_u, :x1max]))
        abs_B_upatmos = [abs_B_upatmos_l, abs_B_upatmos_u]

        abs_B_loatmos_l = np.average(abs(B[:, upatmos_max_l+1:loatmos_max_l, :x1max]))
        abs_B_loatmos_u = np.average(abs(B[:, disk_max_u+1:loatmos_max_u, :x1max]))
        abs_B_loatmos = [abs_B_loatmos_l, abs_B_loatmos_u]

        abs_B_disk_l = np.average(abs(B[:, loatmos_max_l+1:disk_max_l, :x1max]))
        abs_B_disk_u = np.average(abs(B[:, disk_max_l+1:disk_max_u, :x1max]))
        abs_B_disk = [abs_B_disk_l, abs_B_disk_u]

        sign_B_flux_l = np.sum(mf_l) / (2.*np.pi*(x1min**2.))
        sign_B_flux_u = np.sum(mf_u) / (2.*np.pi*(x1min**2.))
        sign_B_flux = [sign_B_flux_l, sign_B_flux_u]

        sign_B_jet_l = np.average(B[:, :jet_max_l, :x1max])
        sign_B_jet_u = np.average(B[:, upatmos_max_u+1:jet_max_u, :x1max])
        sign_B_jet = [sign_B_jet_l, sign_B_jet_u]

        sign_B_upatmos_l = np.average(B[:, jet_max_l+1:upatmos_max_l,:x1max])
        sign_B_upatmos_u = np.average(B[:, loatmos_max_u+1:upatmos_max_u,:x1max])
        sign_B_upatmos = [sign_B_upatmos_l, sign_B_upatmos_u]

        sign_B_loatmos_l = np.average(B[:, upatmos_max_l+1:loatmos_max_l, :x1max])
        sign_B_loatmos_u = np.average(B[:, disk_max_u+1:loatmos_max_u, :x1max])
        sign_B_loatmos = [sign_B_loatmos_l, sign_B_loatmos_u]

        sign_B_disk_l = np.average(B[:, loatmos_max_l+1:disk_max_l, :x1max])
        sign_B_disk_u = np.average(B[:, disk_max_l+1:disk_max_u, :x1max])
        sign_B_disk = [sign_B_disk_l, sign_B_disk_u]

        sim_t = data_cons['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(prob_dir + filename_output, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,
                   abs_B_flux,abs_B_jet,abs_B_upatmos,abs_B_loatmos,abs_B_disk,
                   sign_B_flux,sign_B_jet,sign_B_upatmos,sign_B_loatmos,sign_B_disk]
            writer.writerow(row)



# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate quality factors from simulation data.')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    parser.add_argument('-c', '--component',
                        type=str,
                        default='all',
                        help='specify the B component to calculate (default:all)')
    args = parser.parse_args()

    main(**vars(args))
