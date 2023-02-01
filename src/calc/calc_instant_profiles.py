#!/usr/bin/env python3
#
# calc_instant_profiles.py
#
# A program to calculate radial profiles of various parameters at a specific time in the disk.
#
# Usage: python calc_instant_profiles.py [options]
#
# Output: instant_profiles.csv
#
# Python standard modules
import argparse
import sys
import os
sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
#sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import csv
import numpy as np

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    root = '/home/per29/rds/rds-accretion-zyNhkonJSR8/'
    prob_ids = ['high_res','high_beta','super_res','b200_super_res']
    #times = [110000, 18000, 18050, 110000] # times each file number corresponds to
    file_numbers = ['22000','03600','00280','00550']
    print(os.getcwd())
    os.chdir(root)

    for num in range(0,4): # looping through problem IDs
        t = file_numbers[num]
        prob = prob_ids[num]

        prob_dir = root + prob + '/'
        data_dir = prob_dir + 'data/'
        runfile_dir = prob_dir + 'runfiles/'

        # retrieve lists of scale height with time
        df = pd.read_csv(prob_dir + 'scale_with_time.csv', delimiter='\t', usecols=['sim_time', 'scale_height'])
        scale_time_list = df['sim_time'].to_list()
        scale_height_list = df['scale_height'].to_list()

        data_prim = athena_read.athdf(data_dir + prob + ".prim." + t + ".athdf",
                                        quantities=['press'])
        data_cons = athena_read.athdf(data_dir + prob + ".cons." + t + ".athdf",
                                        quantities=['dens','mom1','mom2','mom3',
                                                    'Bcc1','Bcc2','Bcc3'])

        # find corresponding entry in scale height data
        scale_index = AAT.find_nearest(scale_time_list,data_cons['Time'])
        scale_height = scale_height_list[scale_index]

        #unpack data
        x1v = data_cons['x1v']
        x2v = data_cons['x2v']
        x3v = data_cons['x3v']
        density = data_cons['dens']
        mom1 = data_cons['mom1']
        mom2 = data_cons['mom2']
        mom3 = data_cons['mom3']
        Bcc1 = data_cons['Bcc1']
        Bcc2 = data_cons['Bcc2']
        Bcc3 = data_cons['Bcc3']
        press = data_prim['press']

        # define bounds of region to average over
        th_u = AAT.find_nearest(x2v, np.pi/2. + (3.*scale_height))
        th_l = AAT.find_nearest(x2v, np.pi/2. - (3.*scale_height))

        temp = press/density
        v3 = mom3/density
        GM = 1.

        r,_,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
        orbital_rotation = v3**2./r**2.
        Omega_kep = np.sqrt(GM/(x1v**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(density)[0], np.shape(density)[1], np.shape(density)[2]))

        dmom3 = mom3 - r*Omega_kep

        dens = density[:, th_l:th_u, :]
        mom1 = mom1[:, th_l:th_u, :]
        mom2 = mom2[:, th_l:th_u, :]
        mom3 = mom3[:, th_l:th_u, :]
        temp = temp[:, th_l:th_u, :]
        Bcc1 = Bcc1[:, th_l:th_u, :]
        Bcc2 = Bcc2[:, th_l:th_u, :]
        Bcc3 = Bcc3[:, th_l:th_u, :]
        press = press[:, th_l:th_u, :]
        dmom3 = dmom3[:, th_l:th_u, :]

        stress_Rey = dens*mom1*dmom3
        stress_Max = -Bcc1*Bcc3
        T_rphi = stress_Rey + stress_Max
        alpha_SS = T_rphi/press

        #orbital_rotation = np.average(orbital_rotation, axis=(0,1))
        #Omega_kep = np.average(Omega_kep, axis=(0,1))

        # weight rotation by density
        numWeight_orb = np.sum(orbital_rotation*density) #value * weight
        sum_orb       = np.sum(density) # weight
        weighted_rotation = numWeight_orb/sum_orb
        orbit_v_ratio = weighted_rotation/Omega_kep

        # average over theta and phi
        dens_profile = np.average(dens, axis=(0,1))
        mom1_profile = np.average(mom1, axis=(0,1))
        mom2_profile = np.average(mom2, axis=(0,1))
        mom3_profile = np.average(mom3, axis=(0,1))
        temp_profile = np.average(temp, axis=(0,1))
        Bcc1_profile = np.average(Bcc1, axis=(0,1))
        Bcc2_profile = np.average(Bcc2, axis=(0,1))
        Bcc3_profile = np.average(Bcc3, axis=(0,1))
        orbit_v_ratio_profile = np.average(orbit_v_ratio, axis=(0,1))
        stress_Rey_profile = np.average(stress_Rey, axis=(0,1))
        stress_Max_profile = np.average(stress_Max, axis=(0,1))
        alpha_SS_profile = np.average(alpha_SS, axis=(0,1))

        np.save(prob_dir + 'dens_profile_instant.npy', dens_profile)
        np.save(prob_dir + 'mom1_profile_instant.npy', mom1_profile)
        np.save(prob_dir + 'mom2_profile_instant.npy', mom2_profile)
        np.save(prob_dir + 'mom3_profile_instant.npy', mom3_profile)
        np.save(prob_dir + 'temp_profile_instant.npy', temp_profile)
        np.save(prob_dir + 'Bcc1_profile_instant.npy', Bcc1_profile)
        np.save(prob_dir + 'Bcc2_profile_instant.npy', Bcc2_profile)
        np.save(prob_dir + 'Bcc3_profile_instant.npy', Bcc3_profile)
        np.save(prob_dir + 'rot_profile_instant.npy', orbit_v_ratio_profile)
        np.save(prob_dir + 'Rey_profile_instant.npy', stress_Rey_profile)
        np.save(prob_dir + 'Max_profile_instant.npy', stress_Max_profile)
        np.save(prob_dir + 'alpha_profile_instant.npy', alpha_SS_profile)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate radial profile of various quantities at a specified time from raw simulation data.')
    args = parser.parse_args()

    main(**vars(args))
