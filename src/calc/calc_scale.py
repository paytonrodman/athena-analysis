#!/usr/bin/python
import numpy as np
import os
import sys
#sys.path.insert(0, '/home/per29/athena-public-version-master/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import athena_read
import glob
import re
import csv
import argparse

def main(**kwargs):
    problem  = args.prob_id
    data_dir = "/Users/paytonrodman/athena_sim/" + problem + "/data/"
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open('../scale_with_time.csv', 'r', newline='') as f:
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

    #filename_input = "../run/athinput." + problem
    filename_input = "../athinput." + problem
    data_input = athena_read.athinput(filename_input)
    nx1 = data_input['mesh']['nx1']
    nx2 = data_input['mesh']['nx2']

    scale_height = []
    for t in times:
        print("file number: ", t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + ".cons." + str_t + ".athdf"
        data_cons = athena_read.athdf(filename_cons)

        #unpack data
        x2v = data_cons['x2v'] # theta
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        dens = data_cons['dens']
        # Calculations
        dx1f,dx2f,dx3f = calc_diff(x1f,x2f,x3f)

        up = x2v*dens*np.sin(x2v)*dx2f*dx3f
        down = dens*np.sin(x2v)*dx2f*dx3f
        polar_ang = np.sum(up,axis=(1,2))/np.sum(down,axis=(1,2))

        h_up = (x2v-polar_ang)**2. * dens*np.sin(x2v)*dx2f*dx3f
        h_down = dens*np.sin(x2v)*dx2f*dx3f
        scale_h = np.sqrt(np.sum(h_up,axis=(1,2))/np.sum(h_down,axis=(1,2)))
        scale_h_av = np.average(scale_h,weights=dx1f)

        scale_height.append(scale_h_av)



    times,scale_height = (list(t) for t in zip(*sorted(zip(times,scale_height))))
    os.chdir("../")
    if args.update:
        with open('scale_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,scale_height))
    else:
        with open('scale_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Time", "scale_height"])
            writer.writerows(zip(times,scale_height))

def calc_velocity(mom1,mom2,mom3,vol,dens):
    v1 = mom1*(vol.T)/dens
    v2 = mom2*(vol.T)/dens
    v3 = mom3*(vol.T)/dens
    return v1,v2,v3

def calc_diff(x1f,x2f,x3f):
    dx1f = np.diff(x1f) # delta r
    dx2f = np.diff(x2f) # delta phi
    dx3f = np.diff(x3f) # delta theta
    return dx1f,dx2f,dx3f

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the average geometric scale height over the disk')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('-u', '--update',
                        action="store_true",
                        help='specify whether the results being analysed are from a restart')
    args = parser.parse_args()

    main(**vars(args))
