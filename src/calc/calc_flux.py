#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#sys.path.insert(0, '/home/per29/athena-public-version-master/vis/python')
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import athena_read
import math
import glob
import re
import csv
import scipy.stats
import argparse

def main(**kwargs):
    problem  = args.prob_id
    data_dir = "/Users/paytonrodman/athena_sim/" + problem + "/data/"
    os.chdir(data_dir)

    csv_time = []
    # check if data file already exists
    if args.update:
        with open('../flux_with_time.csv', 'r', newline='') as f:
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
    nx3 = data_input['mesh']['nx3']
    x1min = data_input['mesh']['x1min']
    x1max = data_input['mesh']['x1max']
    x2min = data_input['mesh']['x2min']
    x2max = data_input['mesh']['x2max']
    x3min = data_input['mesh']['x3min']
    x3max = data_input['mesh']['x3max']

    r_id = int(nx1/4.) # middle of high resolution r region
    th_id = int(nx2/2.) # midplane

    mag_flux_u = []
    mag_flux_l = []

    for t in times:
        print("file number: ", t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + ".cons." + str_t + ".athdf"
        data_cons = athena_read.athdf(filename_cons)

        #constants
        gamma = 5./3.
        GM = 1.

        #unpack data
        x1v = data_cons['x1v'] # r
        x2v = data_cons['x2v'] # theta
        x3v = data_cons['x3v'] # phi
        x1f = data_cons['x1f'] # r
        x2f = data_cons['x2f'] # theta
        x3f = data_cons['x3f'] # phi
        Bcc1 = data_cons['Bcc1']

        # Calculations
        dx1f,dx2f,dx3f,vol = calc_volume(x1f,x2f,x3f)

        mf_l = []
        mf_u = []
        for j in np.arange(0,int(len(x2v)/2)):
            for k in np.arange(0,len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_i = Bcc1[k,j,0]*dS
                mf_u.append(mf_i)

        for j in np.arange(int(len(x2v)/2),len(x2v)):
            for k in np.arange(0,len(x3v)):
                dS = (x1f[0]**2.)*np.sin(x2f[j])*dx2f[j]*dx3f[k] # r^2 * sin(theta) * dtheta * dphi
                mf_i = Bcc1[k,j,0]*dS
                mf_l.append(mf_i)

        mag_flux_u.append(np.sum(mf_u))
        mag_flux_l.append(np.sum(mf_l))



    times,mag_flux_u,mag_flux_l = (list(t) for t in zip(*sorted(zip(times,mag_flux_u,mag_flux_l))))
    os.chdir("../")
    if args.update:
        with open('flux_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,mag_flux_u,mag_flux_l))
    else:
        with open('flux_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Time", "mag_flux_u", "mag_flux_l"])
            writer.writerows(zip(times,mag_flux_u,mag_flux_l))


def calc_volume(x1f,x2f,x3f):
    vol = np.empty((len(x1f)-1,len(x2f)-1,len(x3f)-1))
    dx1f = np.diff(x1f) # delta r
    dx2f = np.diff(x2f) # delta theta
    dx3f = np.diff(x3f) # delta phi
    for idx1,x1_len in enumerate(dx1f):
        for idx2,x2_len in enumerate(dx2f):
            for idx3,x3_len in enumerate(dx3f):
                vol[idx1,idx2,idx3] = x1_len*x2_len*x3_len
    return dx1f,dx2f,dx3f,vol

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
