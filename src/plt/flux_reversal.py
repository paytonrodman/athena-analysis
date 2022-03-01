#!/usr/bin/env python3
#
# flux_reversal.py
#
# A program to calculate the time properties of magnetic flux behaviour
#
# To run:
# python flux_reversal.py [options]
#
import argparse
import numpy as np
import os
import sys
sys.path.insert(0, '/Users/paytonrodman/athena/vis/python')
import csv
import matplotlib.pyplot as plt
from matplotlib import colors
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from scipy.optimize import curve_fit

def main(**kwargs):
    # directory containing data
    problem  = args.prob_id
    root_dir = '/Users/paytonrodman/athena-sim/'
    prob_dir = root_dir + problem + '/'
    os.chdir(prob_dir)

    time = []
    mag_flux_u = []
    mag_flux_l = []
    with open('flux_with_time.csv', newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader, None) # skip header
        for row in csv_reader:
            t = float(row[0])
            #t_orb = float(row[1])
            mf_u = float(row[2])
            mf_l = float(row[3])

            time.append(t)
            mag_flux_u.append(mf_u)
            mag_flux_l.append(mf_l)

    time, mag_flux_u, mag_flux_l = zip(*sorted(zip(time, mag_flux_u, mag_flux_l)))

    time = np.asarray(time)
    data = np.asarray(mag_flux_u)

    if problem=='high_res':
        mask = time > 2.e5 #2.e4 (high_beta), 2.e5 (high_res)
    elif problem=='high_beta':
        mask = time > 2.e4 #2.e4 (high_beta), 2.e5 (high_res)
    else:
        mask = time > 0

    time = time[mask]
    data = data[mask]

    #data_fit = savitzky_golay(data_up, 51, 3) # window size 51, polynomial order 3
    #zero_crossings = np.where(np.diff(np.signbit(data_fit)))[0]

    # make data stationary by differencing
    data_diff = np.diff(data)
    data_diff2 = np.diff(data_diff)

    if args.acf:
        autocorrelation_plot(data)
        plt.xlim(-1)
        plt.title('ACF of raw flux data')
        plt.savefig(prob_dir + 'ACF_raw.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff)
        plt.xlim(-1000)
        plt.title('ACF of differenced flux data')
        plt.savefig(prob_dir + 'ACF_diff.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff2)
        plt.xlim(-1000)
        plt.title('ACF of twice differenced flux data')
        plt.savefig(prob_dir + 'ACF_diff2.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff)
        plt.xlim(0,10)
        plt.title('ACF of differenced flux data (zoomed)')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig(prob_dir + 'ACF_diff_zoom.png', dpi=1200)
        plt.close()

        autocorrelation_plot(data_diff2)
        plt.xlim(0,10)
        plt.title('ACF of twice differenced flux data (zoomed)')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig(prob_dir + 'ACF_diff2_zoom.png', dpi=1200)
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
        plt.savefig(prob_dir + 'spectrogram.png', dpi=1200)
        plt.close()

    if args.psd:
        freqs, psd = signal.welch(data)
        psd = (psd-np.nanmin(psd))/(np.nanmax(psd)-np.nanmin(psd))

        new_freqs = np.linspace(np.nanmin(freqs),np.nanmax(freqs),num=10000)

        # fit curve
        popt, _ = curve_fit(DHO_PSD, freqs, psd)
        beta0, beta1, alpha1, alpha2 = popt
        DHO_fit = DHO_PSD(new_freqs, beta0, beta1, alpha1, alpha2)

        popt, _ = curve_fit(DRW_PSD, freqs, psd)
        beta0, alpha1 = popt
        DRW_fit = DRW_PSD(new_freqs, beta0, alpha1)

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
        plt.loglog(new_freqs, DHO_fit, 'k', linestyle=':', label=r'DHO with $\beta_0=${b0:.1e}, $\beta_1=${b1:.1e}, $\alpha_1=${a1:.1e}, $\alpha_2=${a2:.1e}'.format(b0=beta0, b1=beta1, a1=alpha1, a2=alpha2))
        plt.loglog(new_freqs, DRW_fit, 'k', linestyle='-', label=r'DRW with $\beta_0=${b0:.1e}, $\alpha_1=${a1:.1e}'.format(b0=beta02, a1=alpha12))
        plt.loglog(freqs, nu0, 'grey', linestyle='--', label=r'$\propto\nu^{0}$')
        plt.loglog(freqs, nu1, 'orange', linestyle='--', label=r'$\propto\nu^{-1.5}$')
        plt.loglog(freqs, nu2, 'r', linestyle='--', label=r'$\propto\nu^{-2}$')
        plt.title('PSD: power spectral density')
        plt.xlabel(r'Frequency $\nu$')
        plt.ylabel('PSD')
        plt.ylim(1e-4,10)
        plt.xlim(np.nanmin(freqs),np.nanmax(freqs))
        plt.legend()
        plt.tight_layout()
        plt.savefig(prob_dir + 'PSD.png', dpi=1200)
        plt.close()

def DRW_PSD(freq, beta0, alpha1):
    return (beta0**2.) / (alpha1 + (2.*np.pi*freq**2.))

def DHO_PSD(freq, beta0, beta1, alpha1, alpha2):
    return (1. / 2.*np.pi) * ( (beta0**2. + (4.*np.pi**2. * beta1**2. * freq**2.)) / (16.*np.pi**4. * freq**4. + 4.*np.pi**2. * freq**2. *(alpha1**2. - 2.*alpha2) + alpha2**2.) )

def sinusoid(x, amplitude, offset, freq, phase):
    return np.sin(freq*x + phase)*amplitude + offset

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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
    parser.add_argument('--acf',
                        action='store_true',
                        help='whether to plot correlation functions')
    parser.add_argument('--stat',
                        action='store_true',
                        help='whether to test for stationarity')
    parser.add_argument('--spec',
                        action='store_true',
                        help='whether to plot spectrogram')
    parser.add_argument('--psd',
                        action='store_true',
                        help='whether to plot Power Spectral Density (PSD)')
    args = parser.parse_args()

    main(**vars(args))
