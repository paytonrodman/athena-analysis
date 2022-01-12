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
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')
import athena_read
import AAT
import glob
import re
import csv
import argparse
import matplotlib
import matplotlib.pyplot as plt

def main(**kwargs):
    problem  = args.prob_id
    root_dir = "/Users/paytonrodman/athena-sim/"
    #root_dir = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_dir = root_dir + problem + '/'
    data_dir = prob_dir + 'data/'
    runfile_dir = prob_dir + 'runfiles/'
    os.chdir(data_dir)

    csv_time = np.empty(0)
    csv_array = []
    with open(prob_dir + 'flux_with_time.csv', 'r', newline='') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            csv_time = np.append(csv_time, float(row[0]))
            array_entry = np.array([float(row[0]),float(row[2]),float(row[3])]) # [time, mf_u, mf_l]
            csv_array.append(array_entry)

    files = glob.glob('./*.athdf')
    times = np.empty(0)
    for f in files:
        time_sec = re.findall(r'\b\d+\b', f)
        if args.update:
            if float(time_sec[0]) not in times and float(time_sec[0]) in csv_time:
                times = np.append(times, float(time_sec[0]))
        else:
            if float(time_sec[0]) in csv_time:
                times = np.append(times, float(time_sec[0]))
    if len(times)==0:
        sys.exit('No new timesteps to analyse in the given directory. Exiting.')

    #times = [0.,5000.,10000.,15000.,20000.,25000.,30000.]
    #times = [25000.]

    data_input = athena_read.athinput(runfile_dir + 'athinput.' + problem)
    x1min = data_input['mesh']['x1min']
    x1max = data_input['refinement3']['x1max']
    scale_height = data_input['problem']['h_r']
    angles = [0.,1.*scale_height,2.*scale_height,3.*scale_height]
    data_init = athena_read.athdf(problem + '.cons.00000.athdf', quantities=['x2v'])
    x2v = data_init['x2v']
    th_l_id = []
    th_u_id = []
    for val in angles:
        th_u_id.append(AAT.find_nearest(x2v, np.pi/2. + val))
        th_l_id.append(AAT.find_nearest(x2v, np.pi/2. - val))

    Bav = []
    Bcc1_av_0 = []
    Bcc1_av_1 = []
    Bcc1_av_2 = []
    Bcc1_av_3 = []
    sim_time = []
    for t in times:
        print(t)
        str_t = str(int(t)).zfill(5)
        data_cons = athena_read.athdf(problem + '.cons.' + str_t + '.athdf', quantities=['Bcc1','Bcc2','Bcc3'])
        #data_prim = athena_read.athdf(problem + '.prim.' + str_t + '.athdf', quantities=['press'])

        for index,item in enumerate(csv_array):
            if (item[0] == t):
                mf_u = item[1]
                mf_l = item[2]
                continue
        Bav_l = mf_l / (2.*np.pi*(x1min**2.))
        Bav_u = mf_u / (2.*np.pi*(x1min**2.))
        Bav.append([Bav_l,Bav_u])

        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        B = np.sqrt(Bcc1**2. + Bcc2**2. + Bcc3**2.)
        Bcc1_av_0.append(np.average(B[:x1max,th_l_id[0],:]))
        Bcc1_av_1.append([np.average(B[:x1max,th_l_id[1],:]),np.average(B[:x1max,th_u_id[1],:])])
        Bcc1_av_2.append([np.average(B[:x1max,th_l_id[2],:]),np.average(B[:x1max,th_u_id[2],:])])
        Bcc1_av_3.append([np.average(B[:x1max,th_l_id[3],:]),np.average(B[:x1max,th_u_id[3],:])])

        sim_time.append(data_cons['Time'])

    sim_time,Bav,Bcc1_av_0,Bcc1_av_1,Bcc1_av_2,Bcc1_av_3 = (list(t) for t in zip(*sorted(zip(sim_time,Bav,Bcc1_av_0,Bcc1_av_1,Bcc1_av_2,Bcc1_av_3))))
    os.chdir(prob_dir)
    if args.update:
        with open('B_strength_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(sim_time, Bav, Bcc1_av_0, Bcc1_av_1, Bcc1_av_2, Bcc1_av_3))
    else:
        with open('B_strength_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["sim_time", "Bav", "Bcc1_0sh", "B_1sh", "B_2sh", "B_3sh"])
            writer.writerows(zip(sim_time, Bav, Bcc1_av_0, Bcc1_av_1, Bcc1_av_2, Bcc1_av_3))



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
