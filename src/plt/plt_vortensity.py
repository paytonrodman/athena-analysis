#! /usr/bin/env python

"""
Script for plotting vertical (r,theta) or midplane (r,phi) slices of data in
spherical coordinates.

Run "plot_spherical.py -h" to see description of inputs.

See documentation on athena_read.athdf() for important notes about reading files
with mesh refinement.

Users are encouraged to make their own versions of this script for improved
results by adjusting figure size, spacings, tick locations, axes labels, etc.
The script must also be modified to plot any functions of the quantities in the
file, including combinations of multiple quantities.

Requires scipy if making a stream plot.
"""

# Python standard modules
import argparse
import warnings
import sys
#sys.path.insert(0, '/home/per29/rds/rds-accretion-zyNhkonJSR8/athena-analysis/dependencies')
sys.path.insert(0, '/Users/paytonrodman/athena-sim/athena-analysis/dependencies')

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read


# Main function
def main(**kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from scipy.ndimage import map_coordinates

    print('reading data')
    quantities = ['dens','mom1','mom2','mom3']
    if kwargs['b']:
        quantities.append('Bcc1')
        quantities.append('Bcc2')
        quantities.append('Bcc3')
    data = athena_read.athdf(kwargs['data_file'], quantities=quantities)

    # Set resolution of plot in dots per square inch
    if kwargs['dpi'] is not None:
        resolution = kwargs['dpi']

    print('coordinates')
    # Extract basic coordinate information
    coordinates = data['Coordinates']
    r = data['x1v']
    theta = data['x2v']
    phi = data['x3v']
    r_face = data['x1f']
    theta_face = data['x2f']
    phi_face = data['x3f']
    nx1 = len(r)
    nx2 = len(theta)
    nx3 = len(phi)

    # Set radial extent
    if kwargs['r_max'] is not None:
        r_max = kwargs['r_max']
    else:
        r_max = r_face[-1]

    # Account for logarithmic radial coordinate
    if kwargs['logr']:
        r = np.log10(r)
        r_face = np.log10(r_face)
        r_max = np.log10(r_max)

    print('create grid')
    # Create scalar grid
    if kwargs['midplane']:
        phi_grid, theta_grid, r_grid = np.meshgrid(phi_face, theta_face, r_face, indexing='ij') # req all comp for curl later
        # make 2D plotting grid at midplane
        x_grid = r_grid[:, int(nx2/2), :] * np.cos(phi_grid[:, int(nx2/2), :])
        y_grid = r_grid[:, int(nx2/2), :] * np.sin(phi_grid[:, int(nx2/2), :])

    else:
        theta_face_extended = np.concatenate((theta_face, 2.0*np.pi - theta_face[-2::-1]))
        phi_grid, theta_grid, r_grid = np.meshgrid(phi_face, theta_face_extended, r_face, indexing='ij')
        x_grid = r_grid[0, :, :] * np.sin(theta_grid[0, :, :])
        y_grid = r_grid[0, :, :] * np.cos(theta_grid[0, :, :])


    # Create streamline grid
    x_stream = np.linspace(-r_max, r_max, kwargs['stream_samples'])
    if kwargs['midplane']:
        y_stream = np.linspace(-r_max, r_max, kwargs['stream_samples'])
        x_grid_stream, y_grid_stream = np.meshgrid(x_stream, y_stream)
        r_grid_stream_coord = (x_grid_stream.T**2 + y_grid_stream.T**2) ** 0.5
        phi_grid_stream_coord = np.pi + np.arctan2(-y_grid_stream.T, -x_grid_stream.T)
        phi_grid_stream_pix = ((phi_grid_stream_coord + phi[0])
                               / (2.0*np.pi + 2.0 * phi[0])) * (nx3 + 1)
    else:
        z_stream = np.linspace(-r_max, r_max, kwargs['stream_samples'])
        x_grid_stream, z_grid_stream = np.meshgrid(x_stream, z_stream)
        r_grid_stream_coord = (x_grid_stream.T**2 + z_grid_stream.T**2) ** 0.5
        theta_grid_stream_coord = np.pi - \
            np.arctan2(x_grid_stream.T, -z_grid_stream.T)
        if kwargs['theta_compression'] is None:
            theta_grid_stream_pix = ((theta_grid_stream_coord + theta[0])
                                     / (2.0*np.pi + 2.0 * theta[0])) * (2 * nx2 + 1)
        else:
            theta_grid_stream_pix = np.empty_like(theta_grid_stream_coord)
            theta_extended = np.concatenate((-theta[0:1], theta,
                                             2.0*np.pi - theta[::-1],
                                             2.0*np.pi + theta[0:1]))
            for (i, j), theta_val in np.ndenumerate(theta_grid_stream_coord):
                index = sum(theta_extended[1:-1] < theta_val) - 1
                if index < 0:
                    theta_grid_stream_pix[i, j] = -1
                elif index < 2 * nx2 - 1:
                    theta_grid_stream_pix[i, j] = (
                        index + ((theta_val - theta_extended[index])
                                 / (theta_extended[index+1] - theta_extended[index])))
                else:
                    theta_grid_stream_pix[i, j] = 2 * nx2 + 2
    r_grid_stream_pix = np.empty_like(r_grid_stream_coord)
    for (i, j), r_val in np.ndenumerate(r_grid_stream_coord):
        index = sum(r < r_val) - 1
        if index < 0:
            r_grid_stream_pix[i, j] = -1
        elif index < nx1 - 1:
            r_grid_stream_pix[i, j] = index + \
                (r_val - r[index]) / (r[index + 1] - r[index])
        else:
            r_grid_stream_pix[i, j] = nx1

    # Perform slicing/averaging of scalar data
    if kwargs['midplane']:
        if nx2 % 2 == 0:
            vals = np.mean(data[kwargs['quantity']][:, int(nx2/2-1):int(nx2/2+1), :], axis=1)
        else:
            vals = data[kwargs['quantity']][:, int(nx2/2), :]
        if kwargs['average']:
            vals = np.repeat(np.mean(vals, axis=0, keepdims=True), nx3, axis=0)
    else:
        if kwargs['average']:
            vals_right = np.mean(data[kwargs['quantity']], axis=0)
            vals_left = vals_right
        else:
            vals_right = 0.5 * (data[kwargs['quantity']]
                                [-1, :, :] + data[kwargs['quantity']][0, :, :])
            vals_left = 0.5 * (data[kwargs['quantity']][int((nx3/2)-1), :, :]
                               + data[kwargs['quantity']][int(nx3/2), :, :])

    # Join scalar data through boundaries
    if not kwargs['midplane']:
        vals = np.vstack((vals_right, vals_left[::-1, :]))

    # Perform slicing/averaging of vector data
    vr = data['mom1']/data['dens']
    vt = data['mom1']/data['dens']
    vp = data['mom1']/data['dens']
    w1, w2, w3 = curl(r_grid,theta_grid,phi_grid,vr,vt,vp)
    if kwargs['b']: # use magnetovortensity
        vor1 = w1*data['dens']/(data['Bcc1']**2. + data['Bcc2']**2. + data['Bcc3']**2.)
        vor2 = w2*data['dens']/(data['Bcc1']**2. + data['Bcc2']**2. + data['Bcc3']**2.)
        vor3 = w3*data['dens']/(data['Bcc1']**2. + data['Bcc2']**2. + data['Bcc3']**2.)
    else: # use regular vortensity
        vor1 = w1/data['dens']
        vor2 = w2/data['dens']
        vor3 = w3/data['dens']
    if kwargs['midplane']:
        if nx2 % 2 == 0:
            vals_r = np.mean(vor1[:, int(nx2/2-1):int(nx2/2+1), :], axis=1).T
            vals_phi = np.mean(vor3[:, int(nx2/2-1):int(nx2/2+1), :], axis=1).T
        else:
            vals_r = vor1[:, int(nx2/2), :].T
            vals_phi = vor3[:, int(nx2/2), :].T
        if kwargs['stream_average']:
            vals_r = np.tile(np.reshape(np.mean(vals_r, axis=1), (nx1, 1)), nx3)
            vals_phi = np.tile(np.reshape(np.mean(vals_phi, axis=1), (nx1, 1)), nx3)
    else:
        if kwargs['stream_average']:
            vals_r_right = np.mean(vor1, axis=0).T
            vals_r_left = vals_r_right
            vals_theta_right = np.mean(vor2, axis=0).T
            vals_theta_left = -vals_theta_right
        else:
            vals_r_right = vor1[0, :, :].T
            vals_r_left = vor1[int(nx3/2), :, :].T
            vals_theta_right = vor2[0, :, :].T
            vals_theta_left = -vor2[int(nx3/2), :, :].T

    # Join vector data through boundaries
    if kwargs['midplane']:
        vals_r = np.hstack((vals_r[:, -1:], vals_r, vals_r[:, :1]))
        vals_r = map_coordinates(vals_r, (r_grid_stream_pix, phi_grid_stream_pix),
                                 order=1, cval=np.nan)
        vals_phi = np.hstack((vals_phi[:, -1:], vals_phi, vals_phi[:, :1]))
        vals_phi = map_coordinates(vals_phi, (r_grid_stream_pix, phi_grid_stream_pix),
                                   order=1, cval=np.nan)
    else:
        vals_r = np.hstack((vals_r_left[:, :1], vals_r_right, vals_r_left[:, ::-1],
                            vals_r_right[:, :1]))
        vals_r = map_coordinates(vals_r, (r_grid_stream_pix, theta_grid_stream_pix),
                                 order=1, cval=np.nan)
        vals_theta = np.hstack((vals_theta_left[:, :1], vals_theta_right,
                                vals_theta_left[:, ::-1], vals_theta_right[:, :1]))
        vals_theta = map_coordinates(vals_theta,
                                     (r_grid_stream_pix, theta_grid_stream_pix),
                                     order=1, cval=np.nan)

    # Transform vector data to Cartesian components
    if kwargs['logr']:
        r_vals = 10.0**r_grid_stream_coord
        logr_vals = r_grid_stream_coord
    else:
        r_vals = r_grid_stream_coord
    if kwargs['midplane']:
        sin_phi = np.sin(phi_grid_stream_coord)
        cos_phi = np.cos(phi_grid_stream_coord)
        if kwargs['logr']:
            dx_dr = 1.0 / (np.log(10.0) * r_vals) * cos_phi
            dy_dr = 1.0 / (np.log(10.0) * r_vals) * sin_phi
            dx_dphi = -logr_vals * sin_phi
            dy_dphi = logr_vals * cos_phi
        else:
            dx_dr = cos_phi
            dy_dr = sin_phi
            dx_dphi = -r_vals * sin_phi
            dy_dphi = r_vals * cos_phi
        if not (coordinates == 'schwarzschild' or coordinates == 'kerr-schild'):
            dx_dphi /= r_vals
            dy_dphi /= r_vals
        vals_x = dx_dr * vals_r + dx_dphi * vals_phi
        vals_y = dy_dr * vals_r + dy_dphi * vals_phi
    else:
        sin_theta = np.sin(theta_grid_stream_coord)
        cos_theta = np.cos(theta_grid_stream_coord)
        if kwargs['logr']:
            dx_dr = 1.0 / (np.log(10.0) * r_vals) * sin_theta
            dz_dr = 1.0 / (np.log(10.0) * r_vals) * cos_theta
            dx_dtheta = logr_vals * cos_theta
            dz_dtheta = -logr_vals * sin_theta
        else:
            dx_dr = sin_theta
            dz_dr = cos_theta
            dx_dtheta = r_vals * cos_theta
            dz_dtheta = -r_vals * sin_theta
        if not (coordinates == 'schwarzschild' or coordinates == 'kerr-schild'):
            dx_dtheta /= r_vals
            dz_dtheta /= r_vals
        vals_x = dx_dr * vals_r + dx_dtheta * vals_theta
        vals_z = dz_dr * vals_r + dz_dtheta * vals_theta

    # Determine colormapping properties
    cmap = plt.get_cmap(kwargs['colormap'])
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    if kwargs['logc']:
        #norm = colors.LogNorm()
        norm = colors.LogNorm(vmin, vmax)
    else:
        #norm = colors.Normalize()
        norm = colors.Normalize(vmin, vmax)

    # Make plot
    plt.figure()
    #im = plt.pcolormesh(x_grid, y_grid, vals, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    im = plt.pcolormesh(x_grid, y_grid, vals, cmap=cmap, norm=norm)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'invalid value encountered in greater_equal',
            RuntimeWarning,
            'numpy')
        if kwargs['midplane']:
            plt.streamplot(x_stream, y_stream, vals_x.T, vals_y.T,
                           density=kwargs['stream_density'], arrowsize=0.2, linewidth=0.2, color='k')
        else:
            plt.streamplot(x_stream, z_stream, vals_x.T, vals_z.T,
                           density=kwargs['stream_density'], arrowsize=0.2, linewidth=0.2, color='k')
    plt.gca().set_aspect('equal')
    plt.xlim((-r_max, r_max))
    plt.ylim((-r_max, r_max))
    if kwargs['logr']:
        if kwargs['midplane']:
            plt.xlabel(r'$\log_{10}(r)\ x / r$')
            plt.ylabel(r'$\log_{10}(r)\ y / r$')
        else:
            plt.xlabel(r'$\log_{10}(r)\ x / r$')
            plt.ylabel(r'$\log_{10}(r)\ z / r$')
    else:
        if kwargs['midplane']:
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
        else:
            plt.xlabel(r'$x$')
            plt.ylabel(r'$z$')
    if kwargs['time']:
        plt.title(str(int(data['Time'])))
    plt.colorbar(im)
    if kwargs['output_file'] == 'show':
        plt.show()
    else:
        plt.savefig(kwargs['output_file'], bbox_inches='tight', dpi=resolution)


def curl(r,theta,phi,vr,vt,vp):
    dr = r[0,0,:-1]
    dt = theta[0,:-1,0]
    dp = phi[:-1,0,0]

    r_s = r[:-1,:-1,:-1]
    theta_s = theta[:-1,:-1,:-1]
    #phi_s = phi[:-1,:-1,:-1]

    vpsint = vp*np.sin(theta_s) # only works with --midplane
    rvt = r_s*vt
    rvp = r_s*vp

    _, dFr_dt, _ = np.gradient (vr, dr, dt, dp, axis=[2,1,0])
    _, _, dFt_dp = np.gradient (vt, dr, dt, dp, axis=[2,1,0])
    #_, _, _ = np.gradient (vp, dr, dt, dp, axis=[2,1,0])
    _, dFpsint_dt, _ = np.gradient (vpsint, dr, dt, dp, axis=[2,1,0])
    drFt_dr, _, _ = np.gradient (rvt, dr, dt, dp, axis=[2,1,0])
    drFp_dr, _, _ = np.gradient (rvp, dr, dt, dp, axis=[2,1,0])

    rot_r = (1./(r_s*np.sin(theta_s))) * (dFpsint_dt - dFt_dp)
    rot_t = (1./r_s) * ( (1./np.sin(theta_s))*dFr_dt - drFp_dr)
    rot_p = (1./r_s) * (drFt_dr - dFr_dt)

    return rot_r, rot_t, rot_p


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',
                        help='name of input file, possibly including path')
    parser.add_argument('quantity',
                        help='name of quantity to be plotted')
    parser.add_argument('output_file',
                        help=('name of output to be (over)written, possibly including '
                              'path; use "show" to show interactive plot instead'))
    parser.add_argument('-m',
                        '--midplane',
                        action='store_true',
                        help=('flag indicating plot should be midplane (r,phi) rather '
                              'than (r,theta)'))
    parser.add_argument('-a',
                        '--average',
                        action='store_true',
                        help='flag indicating phi-averaging should be done')
    parser.add_argument('-b',
                        '--b',
                        action='store_true',
                        help='flag indicating magnetovortensity should be used')
    parser.add_argument('-l',
                        '--level',
                        type=int,
                        default=None,
                        help=('refinement level to be used in plotting (default: max '
                              'level in file)'))
    parser.add_argument('-r',
                        '--r_max',
                        type=float,
                        default=None,
                        help='maximum radial extent of plot')
    parser.add_argument('--logr',
                        action='store_true',
                        help='flag indicating data should be plotted logarithmically in '
                             'radius')
    parser.add_argument('-c',
                        '--colormap',
                        default=None,
                        help=('name of Matplotlib colormap to use instead of default'))
    parser.add_argument('--vmin',
                        type=float,
                        default=None,
                        help=('data value to correspond to colormap minimum; use '
                              '--vmin=<val> if <val> has negative sign'))
    parser.add_argument('--vmax',
                        type=float,
                        default=None,
                        help=('data value to correspond to colormap maximum; use '
                              '--vmax=<val> if <val> has negative sign'))
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('--stream_average',
                        action='store_true',
                        help='flag indicating phi-averaging on stream plot data')
    parser.add_argument('--stream_density',
                        type=float,
                        default=1.0,
                        help='density of stream lines')
    parser.add_argument('--stream_samples',
                        type=int,
                        default=100,
                        help='linear size of stream line sampling grid')
    parser.add_argument('--theta_compression',
                        type=float,
                        default=None,
                        help=('compression parameter h in '
                              'theta = pi*x_2 + (1-h)/2 * sin(2*pi*x_2)'))
    parser.add_argument('--time',
                        action='store_true',
                        help=('flag indicating title should be set to time'))
    parser.add_argument('--dpi',
                        type=int,
                        default=None,
                        help='resolution of saved image (dots per square inch)')
    args = parser.parse_args()
    main(**vars(args))
