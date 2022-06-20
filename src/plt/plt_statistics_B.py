#!/usr/bin/env python3
#
# plt_statistics_B.py
#
# A program to calculate the time properties of magnetic flux behaviour
#
# To run:
# python plt_statistics_B.py [options]
#
import argparse
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import matplotlib.pyplot as plt
from matplotlib import colors
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from ast import literal_eval

def main(**kwargs):
    if args.component not in ['r','theta','phi','all']:
        sys.exit("Must specify valid B component (r,theta,phi)")

    if args.region not in ['jet','upper_atmos','lower_atmos','disk','all']:
        sys.exit("Must specify valid region for analysis (jet,upper_atmos,lower_atmos,disk,all)")

    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    prob_dir = root_dir + problem + '/'
    os.chdir(prob_dir)

    time = []
    abs_B_flux = []
    abs_B_jet = []
    abs_B_upper = []
    abs_B_lower = []
    abs_B_disk = []
    sign_B_flux = []
    sign_B_jet = []
    sign_B_upper = []
    sign_B_lower = []
    sign_B_disk = []
    with open('B_strength_with_time_'+args.component+'.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            sim_t_i = float(row[0])
            #orbit_t_i = float(row[1])
            abs_b_flux_i = row[2]
            abs_b_jet_i = row[3]
            abs_b_up_i = row[4]
            abs_b_low_i = row[5]
            abs_b_disk_i = row[6]
            sign_b_flux_i = row[7]
            sign_b_jet_i = row[8]
            sign_b_up_i = row[9]
            sign_b_low_i = row[10]
            sign_b_disk_i = row[11]

            time.append(sim_t_i)
            abs_B_flux.append(literal_eval(abs_b_flux_i))
            abs_B_jet.append(literal_eval(abs_b_jet_i))
            abs_B_upper.append(literal_eval(abs_b_up_i))
            abs_B_lower.append(literal_eval(abs_b_low_i))
            abs_B_disk.append(literal_eval(abs_b_disk_i))
            sign_B_flux.append(literal_eval(sign_b_flux_i))
            sign_B_jet.append(literal_eval(sign_b_jet_i))
            sign_B_upper.append(literal_eval(sign_b_up_i))
            sign_B_lower.append(literal_eval(sign_b_low_i))
            sign_B_disk.append(literal_eval(sign_b_disk_i))

    time, abs_B_flux, abs_B_jet, abs_B_upper, abs_B_lower, abs_B_disk, sign_B_flux, sign_B_jet, sign_B_upper, sign_B_lower, sign_B_disk = zip(*sorted(zip(time, abs_B_flux, abs_B_jet, abs_B_upper, abs_B_lower, abs_B_disk, sign_B_flux, sign_B_jet, sign_B_upper, sign_B_lower, sign_B_disk)))

    time = np.asarray(time)
    if args.region == 'all':
        disk = np.asarray(sign_B_disk)
        upper = np.asarray(sign_B_upper)
        lower = np.asarray(sign_B_lower)
        jet = np.asarray(sign_B_jet)
        data = disk + upper + lower + jet
    elif args.region == 'disk':
        data = np.asarray(sign_B_disk)
    elif args.region == 'lower_atmos':
        data = np.asarray(sign_B_lower)
    elif args.region == 'upper_atmos':
        data = np.asarray(sign_B_upper)
    elif args.region == 'jet':
        data = np.asarray(sign_B_jet)

    if problem=='high_res':
        mask = time > 2.e5 #2.e4 (high_beta), 2.e5 (high_res)
    elif problem=='high_beta':
        mask = time > 2.e4 #2.e4 (high_beta), 2.e5 (high_res)
    else:
        mask = time > 0

    time = time[mask]
    data = data[mask]
    data = data[:,0] # choose a hemisphere

    #data_fit = savitzky_golay(data_up, 51, 3) # window size 51, polynomial order 3
    #zero_crossings = np.where(np.diff(np.signbit(data_fit)))[0]

    if args.acf:
        # make data stationary by differencing
        data_diff = np.diff(data)

        autocorrelation_plot(data)
        plt.xlim(-1)
        plt.title('ACF of raw data')
        plt.savefig(prob_dir + args.component + '_' + args.region + '_ACF_raw.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff)
        plt.xlim(-1000)
        plt.title('ACF of differenced data')
        plt.savefig(prob_dir + args.component + '_' + args.region + '_ACF_diff.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff)
        plt.xlim(0,10)
        plt.title('ACF of differenced data (zoomed)')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig(prob_dir + args.component + '_' + args.region + '_ACF_diff_zoom.png', dpi=1200)
        plt.close()

    if args.acf2:
        data_diff2 = np.diff(data_diff)

        autocorrelation_plot(data_diff2)
        plt.xlim(-1000)
        plt.title('ACF of twice differenced data')
        plt.savefig(prob_dir + args.component + '_' + args.region + '_ACF_diff2.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff2)
        plt.xlim(0,10)
        plt.title('ACF of twice differenced data (zoomed)')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig(prob_dir + args.component + '_' + args.region + '_ACF_diff2_zoom.png', dpi=1200)
        plt.close()

    if args.stat:
        result = adfuller(data)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
        	print('\t%s: %.3f' % (key, value))

    if args.spec:
        freqs, _, spectrogram = signal.spectrogram(data)

        plt.figure()
        im = plt.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower', norm=colors.LogNorm())
        plt.colorbar(im)
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        plt.tight_layout()
        plt.savefig(prob_dir + args.component + '_' + args.region + '_spectrogram.png', dpi=1200)
        plt.close()

    if args.psd:
        rg = 6. # r_ISCO
        rav = 100. # approximate average r value
        v_kep = np.sqrt(rg/rav)
        kappa_dict = get_kappa_data()
        if args.region != 'all':
            kappa_arr = np.asarray(kappa_dict[args.region][0])
            kappa_arr = kappa_arr[mask]
            kappa_high = np.max(kappa_arr)
            kappa_low = np.min(kappa_arr)
            kappa_av = np.mean(kappa_arr)
            H = 0.25 # scale height
            tau = (1./kappa_av)*(1./v_kep)*(1./H)
            tau_high = (1./kappa_high)*(1./v_kep)*(1./H)
            tau_low = (1./kappa_low)*(1./v_kep)*(1./H)
            f = 1./tau
            f_high = 1./tau_high
            f_low = 1./tau_low

        freqs, psd = signal.welch(data)
        psd = (psd-np.nanmin(psd))/(np.nanmax(psd)-np.nanmin(psd))

        freqs[freqs==0] = np.nan
        nu0 = freqs**0.
        nu1 = freqs**(-1.5)
        nu1[nu1==np.inf] = np.nan
        nu1 = (nu1-np.nanmin(nu1))/(np.nanmax(nu1)-np.nanmin(nu1))
        nu2 = freqs**(-2.)
        nu2[nu2==np.inf] = np.nan
        nu2 = (nu2-np.nanmin(nu2))/(np.nanmax(nu2)-np.nanmin(nu2))

        plt.figure()
        plt.loglog(freqs, psd, label=r'Normalised data')
        plt.loglog(freqs, nu0, 'grey', linestyle='--', label=r'$\propto\nu^{0}$')
        plt.loglog(freqs, nu1, 'orange', linestyle='--', label=r'$\propto\nu^{-1.5}$')
        plt.loglog(freqs, nu2, 'r', linestyle='--', label=r'$\propto\nu^{-2}$')
        plt.axvline(x=f, color='k', label=r'$1/\tau$={f:.3f}'.format(f=f))
        plt.axvspan(f_high, f_low, alpha=0.5, color='grey')
        #plt.axvline(x=f_high, color='k', linestyle='-.')
        #plt.axvline(x=f_low, color='k', linestyle='-.')
        plt.title('PSD: power spectral density for component "{c}" in region "{r}"'.format(c=args.component, r=args.region))
        plt.xlabel(r'Frequency $\nu$')
        plt.ylabel('PSD')
        plt.ylim(1e-5,1e1)
        plt.xlim(np.nanmin(freqs),np.nanmax(freqs))
        plt.legend()
        plt.tight_layout()
        plt.savefig(prob_dir + args.component + '_' + args.region + '_PSD.png', dpi=1200)
        plt.close()

def get_kappa_data():
    import pandas as pd
    import numpy as np

    time2 = []
    k_jet = []
    k_upatmos = []
    k_lowatmos = []
    k_disk = []
    with open('curvature_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            jet = np.fromstring(row[2].strip("[]"), sep=',')
            upatmos = np.fromstring(row[3].strip("[]"), sep=',')
            lowatmos = np.fromstring(row[4].strip("[]"), sep=',')
            disk = np.fromstring(row[5].strip("[]"), sep=',')

            time2.append(t)
            k_jet.append(jet)
            k_upatmos.append(upatmos)
            k_lowatmos.append(lowatmos)
            k_disk.append(disk)

    time2, k_jet, k_upatmos, k_lowatmos, k_disk = zip(*sorted(zip(time2, k_jet, k_upatmos, k_lowatmos, k_disk)))
    k_jet = np.asarray(k_jet)
    k_upatmos = np.asarray(k_upatmos)
    k_lowatmos = np.asarray(k_lowatmos)
    k_disk = np.asarray(k_disk)

    #window_size = 100
    #av_k_jet = [pd.Series(k_jet[:,0]).rolling(window_size).mean(),
    #            pd.Series(k_jet[:,1]).rolling(window_size).mean()]
    #av_k_upatmos = [pd.Series(k_upatmos[:,0]).rolling(window_size).mean(),
    #                pd.Series(k_upatmos[:,1]).rolling(window_size).mean()]
    #av_k_lowatmos = [pd.Series(k_lowatmos[:,0]).rolling(window_size).mean(),
    #                 pd.Series(k_lowatmos[:,1]).rolling(window_size).mean()]
    #av_k_disk = [pd.Series(k_disk[:,0]).rolling(window_size).mean(),
    #             pd.Series(k_disk[:,1]).rolling(window_size).mean()]

    kappa_dict = {'disk': k_disk.T,
                  'lower_atmos': k_lowatmos.T,
                  'upper_atmos': k_upatmos.T,
                  'jet': k_jet.T}

    return kappa_dict

def DRW_PSD(freq, beta0, alpha1):
    import numpy as np
    return (beta0**2.) / (alpha1 + (2.*np.pi*freq**2.))

def DHO_PSD(freq, beta0, beta1, alpha1, alpha2):
    import numpy as np

    return (1. / 2.*np.pi) * ( (beta0**2. + (4.*np.pi**2. * beta1**2. * freq**2.)) / (16.*np.pi**4. * freq**4. + 4.*np.pi**2. * freq**2. *(alpha1**2. - 2.*alpha2) + alpha2**2.) )

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average magnetic field values within the midplane at a given radius (r)')
    parser.add_argument('prob_id',
                        help='base name of the data being analysed, e.g. inflow_var or disk_base')
    parser.add_argument('component',
                        help='the magnetic field component being studied, e.g. r, theta, phi, or all')
    parser.add_argument('region',
                        help='the region being studied, e.g. disk, lower_atmos, upper_atmos, jet, or all')
    parser.add_argument('--acf',
                        action='store_true',
                        help='plot correlation functions to first difference')
    parser.add_argument('--acf2',
                        action='store_true',
                        help='plot correlation function to second difference')
    parser.add_argument('--stat',
                        action='store_true',
                        help='test for stationarity')
    parser.add_argument('--spec',
                        action='store_true',
                        help='plot spectrogram')
    parser.add_argument('--psd',
                        action='store_true',
                        help='plot Power Spectral Density (PSD)')
    args = parser.parse_args()

    main(**vars(args))
