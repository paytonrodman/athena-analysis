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
        with open('../qual_with_time.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f, delimiter='\t')
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

    theta_B = []
    av_Q_theta = []
    min_Q_theta = []
    max_Q_theta = []
    av_Q_phi = []
    min_Q_phi = []
    max_Q_phi = []

    prim_var_names = ['press']
    for t in times:
        print("file number: ", t)
        str_t = str(int(t)).zfill(5)

        filename_cons = problem + ".cons." + str_t + ".athdf"
        filename_prim = problem + ".prim." + str_t + ".athdf"

        data_prim = athena_read.athdf(filename_prim)
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
        dens = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        # Calculations
        dx1f,dx2f,dx3f,vol = calc_volume(x1f,x2f,x3f)
        v1,v2,v3 = calc_velocity(mom1,mom2,mom3,vol,dens)
        Omega_kep = np.sqrt(GM/(x1v**3.)) #Keplerian angular velocity in midplane

        if args.Qtheta or args.Qphi:
            Qt,Qp = quality_factors(x1v,x3v,dx2f,dx3f,
                                    dens,press,v2,
                                    Bcc1,Bcc2,Bcc3,
                                    Omega_kep,gamma,th_id)
            print(Qt)
            print(wbdjw)
        if args.Qtheta:
            m_Q_theta,l_Q_theta,u_Q_theta = mean_confidence_interval(Qt.flatten(), confidence=0.95)
        else:
            m_Q_theta = np.nan
            l_Q_theta = np.nan
            u_Q_theta = np.nan
        if args.Qphi:
            m_Q_phi,l_Q_phi,u_Q_phi = mean_confidence_interval(Qp.flatten(), confidence=0.95)
        else:
            m_Q_phi = np.nan
            l_Q_phi = np.nan
            u_Q_phi = np.nan
        av_Q_theta.append(m_Q_theta)
        min_Q_theta.append(l_Q_theta)
        max_Q_theta.append(u_Q_theta)
        av_Q_phi.append(m_Q_phi)
        min_Q_phi.append(l_Q_phi)
        max_Q_phi.append(u_Q_phi)
        print(av_Q_theta)

        tB = magnetic_angle(Bcc1,Bcc2,th_id)
        theta_B.append(tB)
        #M_rphi,R_rphi = stress_tensors()

        #X.append(Bcc3_av)
        #Y.append(ep3_av)

    times,theta_B,av_Q_theta,min_Q_theta,max_Q_theta,av_Q_phi,min_Q_phi,max_Q_phi = (list(t) for t in zip(*sorted(zip(times,theta_B,av_Q_theta,min_Q_theta,max_Q_theta,av_Q_phi,min_Q_phi,max_Q_phi))))
    os.chdir("../")
    if args.update:
        with open('qual_with_time.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,theta_B,av_Q_theta,min_Q_theta,max_Q_theta,av_Q_phi,min_Q_phi,max_Q_phi))
    else:
        with open('qual_with_time.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(times,theta_B,av_Q_theta,min_Q_theta,max_Q_theta,av_Q_phi,min_Q_phi,max_Q_phi))


def magnetic_angle(Bcc1,Bcc2,th_id):
    theta_B = math.degrees(-np.arctan(np.mean(Bcc1[:,th_id,:]/Bcc2[:,th_id,:])))
    return theta_B

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def quality_factors(x1v,x3v,dx2f,dx3f,dens,press,v2,Bcc1,Bcc2,Bcc3,Omega_kep,gamma,th_id):
    vA_theta,vA_phi = Alfven_vel(dens,press,Bcc1,Bcc2,Bcc3,gamma,th_id)
    lambda_MRI_theta = 2.*np.pi*np.sqrt(16/15)*np.abs(vA_theta)/Omega_kep
    lambda_MRI_phi = 2.*np.pi*np.sqrt(16/15)*np.abs(vA_phi)/Omega_kep
    delta_vphi = v2[:,th_id,:] - x1v*Omega_kep
    Q_theta = lambda_MRI_theta/np.sqrt(x1v*dx3f)
    Q_phi = lambda_MRI_phi/np.sqrt(x1v*np.abs(np.sin(x3v))*dx2f)
    return Q_theta,Q_phi

def Alfven_vel(dens,press,Bcc1,Bcc2,Bcc3,gamma,th_id):
    w = dens[:,th_id,:] + (gamma/(gamma - 1.))*press[:,th_id,:]
    B2 = Bcc1[:,th_id,:]**2 + Bcc2[:,th_id,:]**2 + Bcc3[:,th_id,:]**2
    vA_theta = Bcc3[:,th_id,:]/(np.sqrt(w+B2)) #Alfven velocity of theta component of B
    vA_phi = Bcc2[:,th_id,:]/(np.sqrt(w+B2)) #Alfven velocity of phi component of B
    return vA_theta,vA_phi

def calc_velocity(mom1,mom2,mom3,vol,dens):
    v1 = mom1*(vol.T)/dens
    v2 = mom2*(vol.T)/dens
    v3 = mom3*(vol.T)/dens
    return v1,v2,v3

def calc_volume(x1f,x2f,x3f):
    vol = np.empty((len(x1f)-1,len(x2f)-1,len(x3f)-1))
    dx1f = np.diff(x1f) # delta r
    dx2f = np.diff(x2f) # delta phi
    dx3f = np.diff(x3f) # delta theta
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
    parser.add_argument('-Qt', '--Qtheta',
                        action="store_true",
                        help='calculate theta Quality factor')
    parser.add_argument('-Qp', '--Qphi',
                        action="store_true",
                        help='calculate phi Quality factor')
    args = parser.parse_args()

    main(**vars(args))
