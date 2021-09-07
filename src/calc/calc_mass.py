#!/usr/bin/python
import numpy as np
import os
import sys
sys.path.insert(0, '../../dependencies')
import athena_read
from AAT import calculate_delta,calculate_velocity
import glob
import re
import csv
import argparse

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena_sim/"
    data_dir = root_dir + problem + "/data/"
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open('../mass_with_time.csv', 'r', newline='') as f:
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

    mf_total = []
    for t in sorted(times):
        print("file number: ", t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + ".cons." + str_t + ".athdf"
        data_cons = athena_read.athdf(filename_cons)

        #unpack data
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        # Calculations
        dx1f,dx2f,dx3f = calculate_delta(x1f,x2f,x3f)
        v1,v2,v3 = calculate_velocity(mom1,mom2,mom3,dens)

        mf = []
        for j in np.arange(0,len(x2v)):
            for k in np.arange(0,len(x3v)):
                dOmega = np.sin(x2f[j]) * dx2f[j] * dx3f[k]
                mf_i = -dens[k,j,0] * v1[k,j,0] * (x1f[0])**2. * dOmega
                mf.append(mf_i)

        mf_total.append(np.sum(mf))


    times,mf_total = (list(t) for t in zip(*sorted(zip(times,mf_total))))
    os.chdir("../")
    if args.update:
        with open('mass_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,mf_total))
    else:
        with open('mass_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Time", "mass_flux"])
            writer.writerows(zip(times,mf_total))

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate instantaneous mass flux across inner radial boundary')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
