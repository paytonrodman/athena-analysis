#!/usr/bin/env python3
#
# plt_alpha.py
#
# A program to plot a 2D map of the azimuthally-averaged Shakura-Sunyaev alpha.
#
# Usage: python plt_alpha.py [options]
#
# Python standard modules
import argparse
import sys
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read
import AAT

def main(**kwargs):
    # Load Python plotting modules
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Set resolution of plot in dots per square inch
    if args.dpi is not None:
        resolution = args.dpi

    data_input = athena_read.athinput(args.input)
    scale_height = data_input['problem']['h_r']

    if 'cons' in args.data_file:
        file_cons = args.data_file
        file_prim = args.data_file.replace('cons','prim')
    else:
        file_prim = args.data_file
        file_cons = args.data_file.replace('prim','cons')

    data_prim = athena_read.athdf(file_prim, quantities=['press'])
    data_cons = athena_read.athdf(file_cons, quantities=['x1v','x2v','x3v',
                                                        'x1f','x3f',
                                                        'dens',
                                                        'mom1','mom3',
                                                        'Bcc1','Bcc3'])

    #unpack data
    x1v = data_cons['x1v']
    x2v = data_cons['x2v']
    x3v = data_cons['x3v']
    x1f = data_cons['x1f']
    x3f = data_cons['x3f']
    dens = data_cons['dens']
    mom1 = data_cons['mom1']
    mom3 = data_cons['mom3']
    Bcc1 = data_cons['Bcc1']
    Bcc3 = data_cons['Bcc3']
    press = data_prim['press']

    th_u = AAT.find_nearest(x2v, np.pi/2. + (2.*scale_height))
    th_l = AAT.find_nearest(x2v, np.pi/2. - (2.*scale_height))

    r,_,_ = np.meshgrid(x3v,x2v,x1v, sparse=False, indexing='ij')
    GM = 1.
    Omega_kep = np.sqrt(GM/(x1v**3.))
    Omega_kep = np.broadcast_to(Omega_kep, (np.shape(dens)[0], np.shape(dens)[1], np.shape(dens)[2]))
    dmom3 = mom3 - r*Omega_kep

    press = press[:, th_l:th_u, :]
    dens = dens[:, th_l:th_u, :]
    mom1 = mom1[:, th_l:th_u, :]
    Bcc1 = Bcc1[:, th_l:th_u, :]
    Bcc3 = Bcc3[:, th_l:th_u, :]
    dmom3 = dmom3[:, th_l:th_u, :]

    #plt.pcolormesh(np.average(dmom3, axis=(1)), cmap="viridis", shading='auto')

    Reynolds_stress = dens*mom1*dmom3
    Maxwell_stress = -Bcc1*Bcc3/(4.*np.pi)

    T_rphi = Reynolds_stress + Maxwell_stress
    T_rphi = np.average(T_rphi, axis=(1)) # average over vertical height, theta
    T_mag = Maxwell_stress
    T_mag = np.average(T_mag, axis=(1)) # average over vertical height, theta

    alpha_SS = T_rphi/np.average(press, axis=(1))
    alpha_mag = T_mag/np.average(press, axis=(1))

    # Create scalar grid
    r_grid, phi_grid = np.meshgrid(x1f, x3f)
    x_grid = r_grid * np.cos(phi_grid)
    y_grid = r_grid * np.sin(phi_grid)

    vals = [T_rphi,alpha_SS,alpha_mag]
    labels = [r'$\langle T_{r\phi}\rangle_{\theta}$', r'$\langle \alpha_{SS}\rangle_{\theta}$', r'$\langle \alpha_{\rm mag}\rangle_{\theta}$']
    outputs = ['Trphi.png', 'alpha.png', 'alpha_mag.png']
    for num in np.arange(0,3):
        data = vals[num]
        fig = plt.figure()
        fig.add_subplot(111)
        if args.vmin is not None:
            vmin = args.vmin
        else:
            vmin = np.nanmin(data[np.isfinite(data)])
        if args.vmax is not None:
            vmax = args.vmax
        else:
            vmax = np.nanmax(data[np.isfinite(data)])

        if num in [1,2] and args.logc:
            norm = colors.SymLogNorm(linthresh=1e-5, linscale=1e-5,
                                          vmin=vmin, vmax=vmax, base=10)
        elif num==0 and args.logc:
            vmin = 1e-10
            norm = colors.LogNorm(vmin, vmax)
        else:
            norm = colors.Normalize(vmin, vmax)
        im = plt.pcolormesh(x_grid, y_grid, data, cmap="viridis", norm=norm, shading='auto')
        cbar = plt.colorbar(im, extend='both')
        plt.gca().set_aspect('equal')
        cbar.set_label(labels[num])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('t={:.2e} GM/c3'.format(data_cons['Time']))
        plt.savefig(args.output_location + outputs[num], bbox_inches='tight', dpi=resolution)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate various quality factors from raw simulation data.')
    parser.add_argument('data_file',
                        help='name of the .athdf data file to be analysed, including path')
    parser.add_argument('input',
                        help='location of athinput file, including path')
    parser.add_argument('output_location',
                        help='folder to save outputs to, including path')
    parser.add_argument('-a', '--average',
                        action='store_true',
                        help='flag indicating phi-averaging should be done')
    parser.add_argument('-r', '--r_max',
                        type=float,
                        default=None,
                        help='maximum radial extent of plot')
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('--dpi',
                        type=int,
                        default=200,
                        help='resolution of saved image (dots per square inch)')
    parser.add_argument('--vmin',
                        type=float,
                        default=None,
                        help='minimum value for colormap')
    parser.add_argument('--vmax',
                        type=float,
                        default=None,
                        help='maximum value for colormap')
    args = parser.parse_args()

    main(**vars(args))
