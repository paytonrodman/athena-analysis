#!/usr/bin/env python3
#
# calc_curvature.py
#
# A program to calculate the curvature, kappa, of magnetic field lines in the r-theta plane.
#
# Usage: mpirun -n [nprocs] calc_curvature.py [options]
#
# Python standard modules
import argparse
import warnings
import sys
import os
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')

# Other Python modules
import numpy as np
from mpi4py import MPI
import csv

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    if kwargs['stream'] is None:
        sys.exit('Must specify stream')

    # get number of processors and processor rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    os.chdir(kwargs['data'])

    file_times = AAT.add_time_to_list(kwargs['update'], kwargs['output'])
    local_times = AAT.distribute_files_to_cores(file_times, size, rank)

    # Determine if vector quantities should be read
    quantities = []
    quantities.append(kwargs['stream'] + '1')
    quantities.append(kwargs['stream'] + '2')

    data_input = athena_read.athinput(kwargs['input'])
    if 'refinement3' not in data_input:
        sys.exit('Simulation must have 3 levels of refinement in mesh. Exiting.')
    x1max = data_input['refinement3']['x1max']

    data_init = athena_read.athdf(kwargs['problem_id'] + '.cons.00000.athdf', quantities=['x2v'])
    x2v = data_init['x2v']

    jet_max_l = AAT.find_nearest(x2v, data_input['refinement1']['x2min'])
    upatmos_max_l = AAT.find_nearest(x2v, data_input['refinement2']['x2min'])
    loatmos_max_l = AAT.find_nearest(x2v, data_input['refinement3']['x2min'])
    disk_max_l = AAT.find_nearest(x2v, np.pi/2.)

    disk_max_u = AAT.find_nearest(x2v, data_input['refinement3']['x2max'])
    loatmos_max_u = AAT.find_nearest(x2v, data_input['refinement2']['x2max'])
    upatmos_max_u = AAT.find_nearest(x2v, data_input['refinement1']['x2max'])
    jet_max_u = AAT.find_nearest(x2v, np.pi)

    if rank==0:
        if not kwargs['update']:
            with open(kwargs['output'], 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["sim_time", "orbit_time",
                "kappa_jet", "kappa_upatmos", "kappa_lowatmos", "kappa_disk"])
    for t in local_times:
        str_t = str(int(t)).zfill(5)
        data = athena_read.athdf(kwargs['problem_id'] + '.cons.' + str_t + '.athdf',
                                    quantities=quantities)

        # Extract basic coordinate information
        r = data['x1v']
        theta = data['x2v']
        phi = data['x3v']
        r_face = data['x1f']
        theta_face = data['x2f']
        nx3 = len(phi)

        # Create scalar grid
        theta_face_extended = np.concatenate((theta_face, 2.0*np.pi - theta_face[-2::-1]))
        r_grid, theta_grid = np.meshgrid(r_face, theta_face_extended)
        x_grid = r_grid * np.sin(theta_grid)
        y_grid = r_grid * np.cos(theta_grid)

        # Perform slicing/averaging of vector data
        vals_r_right = data[kwargs['stream'] + '1'][0, :, :].T
        vals_r_left = data[kwargs['stream'] + '1'][int(nx3/2), :, :].T
        vals_theta_right = data[kwargs['stream'] + '2'][0, :, :].T
        vals_theta_left = -data[kwargs['stream'] + '2'][int(nx3/2), :, :].T
        # Join vector data through boundaries
        vals_r = np.hstack((vals_r_left[:, :1], vals_r_right[:, :-1],
                            vals_r_left[:, -2::-1], vals_r_right[:, :1]))
        vals_theta = np.hstack((vals_theta_left[:, :1], vals_theta_right[:, :-1],
                                vals_theta_left[:, -2::-1], vals_theta_right[:, :1]))

        # Transform vector data to Cartesian components
        r_vals = r_grid[:-1, :-1]
        sin_theta = np.sin(theta_grid[:-1, :-1])
        cos_theta = np.cos(theta_grid[:-1, :-1])
        dx_dr = sin_theta
        dz_dr = cos_theta
        dx_dtheta = (r_vals * cos_theta) / r_vals
        dz_dtheta = (-r_vals * sin_theta) / r_vals

        vals_x = dx_dr.T * vals_r + dx_dtheta.T * vals_theta
        vals_z = dz_dr.T * vals_r + dz_dtheta.T * vals_theta

        # curvature kappa = ||b dot nabla b|| where b = B / ||B|| and ||x|| is the modulus of x
        # and nabla is the differential operator, i d/dx + j d/dy + k d/dz
        bx = vals_x.T / np.sqrt(vals_x.T**2. + vals_z.T**2.)
        bz = vals_z.T / np.sqrt(vals_x.T**2. + vals_z.T**2.)

        dbx = np.array(np.gradient(bx))
        dx = np.array(np.gradient(x_grid))
        dbz = np.array(np.gradient(bz))
        dz = np.array(np.gradient(y_grid))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=RuntimeWarning)
            divbx = np.true_divide(dbx[1,:,:], dx[1,:-1,:-1])
            divbz = np.true_divide(dbz[0,:,:], dz[0,:-1,:-1])

        bdivb = divbx*bx + divbz*bz
        bdivb = np.abs(bdivb)

        # Calculate average curvature in each region
        curve_jet_l = bdivb[:jet_max_l, :x1max]
        curve_jet_u = bdivb[upatmos_max_u+1:jet_max_u, :x1max]
        curve_jet = [np.median(curve_jet_l[np.isfinite(curve_jet_l)]), np.median(curve_jet_u[np.isfinite(curve_jet_u)])]

        curve_upatmos_l = bdivb[jet_max_l+1:upatmos_max_l,:x1max]
        curve_upatmos_u = bdivb[loatmos_max_u+1:upatmos_max_u,:x1max]
        curve_upatmos = [np.median(curve_upatmos_l[np.isfinite(curve_upatmos_l)]), np.median(curve_upatmos_u[np.isfinite(curve_upatmos_u)])]

        curve_loatmos_l = bdivb[upatmos_max_l+1:loatmos_max_l, :x1max]
        curve_loatmos_u = bdivb[disk_max_u+1:loatmos_max_u, :x1max]
        curve_loatmos = [np.median(curve_loatmos_l[np.isfinite(curve_loatmos_l)]), np.median(curve_loatmos_u[np.isfinite(curve_loatmos_u)])]

        curve_disk_l = bdivb[loatmos_max_l+1:disk_max_l, :x1max]
        curve_disk_u = bdivb[disk_max_l+1:disk_max_u, :x1max]
        curve_disk = [np.median(curve_disk_l[np.isfinite(curve_disk_l)]), np.median(curve_disk_u[np.isfinite(curve_disk_u)])]

        sim_t = data['Time']
        orbit_t = AAT.calculate_orbit_time(sim_t)

        with open(kwargs['output'], 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [sim_t,orbit_t,
                   curve_jet,curve_upatmos,curve_loatmos,curve_disk]
            writer.writerow(row)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate quality factors from simulation data.')
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
    parser.add_argument('-s', '--stream',
                        default=None,
                        help='name of vector quantity to use to make stream plot')
    args = parser.parse_args()

    main(**vars(args))
