#!/usr/bin/python
#
# Calculate magnetic flux over innermost upper and lower hemispheres
#
# Usage: python calc_flux.py [problem_id] [-u]
#
import numpy as np
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
from math import sqrt
import argparse

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open(prob_dir + 'flux_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader, None) # skip header
            for row in csv_reader:
                csv_time.append(float(row[0]))

    files = glob.glob('./*.athdf')
    times = []
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) not in csv_time:
                times.append(float(time_sec[0]))
        else:
            if float(time_sec[0]) not in times:
                times.append(float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')


    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x2v'])
    x2v_init = data_init['x2v']
    th_id = AAT.find_nearest(x2v_init, np.pi/2.)

    orbit_time = []
    sim_time = []
    mag_flux_u = []
    mag_flux_l = []
    for t in sorted(times):
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['x2v','x3v','x1f','x2f','x3f','Bcc1'])

        #unpack data
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        Bcc1 = data_cons['Bcc1']

        # Calculations
        _,dx2f,dx3f = AAT.calculate_delta(x1f,x2f,x3f)

        mf_l = []
        mf_u = []

        for j in range(th_id):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_i = Bcc1[k,j,0]*dS
                mf_u.append(mf_i)

        for j in range(th_id,len(x2v)):
            for k in range(len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_i = Bcc1[k,j,0]*dS
                mf_l.append(mf_i)

        mag_flux_u.append(np.sum(mf_u))
        mag_flux_l.append(np.sum(mf_l))

        r_ISCO = 6. # location of ISCO in PW potential
        T_period = 2.*np.pi*sqrt(r_ISCO)*(r_ISCO - 2.)
        sim_t.append(data_cons['Time'])
        orbit_time.append(sim_t/T_period)

    sim_time,orbit_time,mag_flux_u,mag_flux_l = (list(t) for t in zip(*sorted(zip(sim_time,orbit_time,mag_flux_u,mag_flux_l))))
    os.chdir(prob_dir)
    if args.update:
        with open('flux_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(sim_time,orbit_time,mag_flux_u,mag_flux_l))
    else:
        with open('flux_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["sim_time", "orbit_time", "mag_flux_u", "mag_flux_l"])
            writer.writerows(zip(sim_time,orbit_time,mag_flux_u,mag_flux_l))

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
