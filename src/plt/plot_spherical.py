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
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector

# Athena++ modules
import athena_read
import AAT


# Main function
def main(**kwargs):

    # Load function for transforming coordinates
    if kwargs['stream'] is not None:
        from scipy.ndimage import map_coordinates

    # Load Python plotting modules
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.offsetbox import AnchoredText

    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    matplotlib.use('pdf')
    #plt.rcParams['axes.facecolor'] = 'black'

    if kwargs['quantity']=='dens':
        cbar_title = r'$\rho$'
    elif kwargs['quantity']=='beta':
        cbar_title = r'$\beta$'
    elif kwargs['quantity']=='alfven':
        cbar_title = r'$v_A$'

    # Determine if vector quantities should be read
    if kwargs['quantity']=='beta':
        quantities = ['dens','Bcc1','Bcc2','Bcc3'] # quantities needed to calculate beta
        prim_quantities = ['press']
    elif kwargs['quantity']=='alfven':
        quantities = ['dens','Bcc1','Bcc2','Bcc3'] # quantities needed to calculate the Alfven speed
    else:
        quantities = [kwargs['quantity']]

    if kwargs['stream'] is not None:
        if kwargs['stream']=='v':
            quantities.append('mom1')
        else:
            quantities.append(kwargs['stream'] + '1')
        if kwargs['midplane']:
            if kwargs['stream']=='v':
                quantities.append('mom3')
            else:
                quantities.append(kwargs['stream'] + '3')
        else:
            if kwargs['stream']=='v':
                quantities.append('mom2')
            else:
                quantities.append(kwargs['stream'] + '2')

    n = len(kwargs['data_file'])
    fig, axs = plt.subplots(nrows=2, ncols=n, figsize=(n*4, 8))

    grid_X = [[] for _ in range(n)]
    grid_Y = [[] for _ in range(n)]
    bg = [[] for _ in range(n)]
    stream_X = [[] for _ in range(n)]
    stream_Y = [[] for _ in range(n)]
    vals_X = [[] for _ in range(n)]
    vals_Y = [[] for _ in range(n)]
    titles = []
    labels = []
    # Read data
    for idx in range(n):
        f = kwargs['data_file'][idx]
        file = f[f.rindex('/'):][1:]
        prob_id = file[:file.index('.')]

        l,_,_ = AAT.problem_dictionary(prob_id, False)
        labels.append(l)

        data = athena_read.athdf(f, quantities=quantities)
        if kwargs['quantity']=='beta':
            prim_f = f.replace("cons", "prim")
            data_p = athena_read.athdf(prim_f, quantities=prim_quantities)

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
        titles.append(int(data['Time']))

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

        # Create scalar grid
        if kwargs['midplane']:
            r_grid, phi_grid = np.meshgrid(r_face, phi_face)
            x_grid = r_grid * np.cos(phi_grid)
            y_grid = r_grid * np.sin(phi_grid)
        else:
            theta_face_extended = np.concatenate((theta_face, 2.0*np.pi - theta_face[-2::-1]))
            r_grid, theta_grid = np.meshgrid(r_face, theta_face_extended)
            x_grid = r_grid * np.sin(theta_grid)
            y_grid = r_grid * np.cos(theta_grid)

        # Create streamline grid
        if kwargs['stream'] is not None:
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
                theta_grid_stream_pix = ((theta_grid_stream_coord + theta[0])
                                         / (2.0*np.pi + 2.0 * theta[0])) * (2 * nx2 + 1)
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
        if kwargs['quantity']=='beta':
            # beta = P_g / P_b
            # P_b = B^2 / 2mu_0
            P_b = (data['Bcc1']**2. + data['Bcc3']**2. + data['Bcc3']**2.) / 2.
            scalar_q = data_p['press']/P_b
        elif kwargs['quantity']=='alfven':
            # v_A = B / sqrt(rho)
            scalar_q = np.sqrt(data['Bcc1']**2. + data['Bcc3']**2. + data['Bcc3']**2.)/np.sqrt(data['dens'])
        else:
            scalar_q = data[kwargs['quantity']]
        if kwargs['midplane']:
            if nx2 % 2 == 0:
                vals = np.mean(scalar_q[:, int(nx2/2-1):int(nx2/2+1), :], axis=1)
            else:
                vals = scalar_q[:, int(nx2/2), :]
            if kwargs['average']:
                vals = np.repeat(np.mean(vals, axis=0, keepdims=True), nx3, axis=0)
        else:
            if kwargs['average']:
                vals_right = np.mean(scalar_q, axis=0)
                vals_left = vals_right
            else:
                vals_right = 0.5 * (scalar_q[-1, :, :] + scalar_q[0, :, :])
                vals_left = 0.5 * (scalar_q[int((nx3/2)-1), :, :] + scalar_q[int(nx3/2), :, :])

        # Join scalar data through boundaries
        if not kwargs['midplane']:
            vals = np.vstack((vals_right, vals_left[::-1, :]))

        # Perform slicing/averaging of vector data
        if kwargs['stream'] is not None:
            if kwargs['midplane']:
                if kwargs['stream']=='v':
                    stream1 = data['mom1']*data['dens']
                    stream3 = data['mom3']*data['dens']
                else:
                    stream1 = data[kwargs['stream'] + '1']
                    stream3 = data[kwargs['stream'] + '3']

                if nx2 % 2 == 0:
                    vals_r = np.mean(stream1[:, int(nx2/2-1):int(nx2/2+1), :], axis=1).T
                    vals_phi = np.mean(stream3[:, int(nx2/2-1):int(nx2/2+1), :], axis=1).T
                else:
                    vals_r = stream1[:, int(nx2/2), :].T
                    vals_phi = stream3[:, int(nx2/2), :].T
                if kwargs['stream_average']:
                    vals_r = np.tile(np.reshape(np.mean(vals_r, axis=1), (nx1, 1)), nx3)
                    vals_phi = np.tile(np.reshape(np.mean(vals_phi, axis=1), (nx1, 1)), nx3)
            else:
                if kwargs['stream']=='v':
                    stream1 = data['mom1']*data['dens']
                    stream2 = data['mom2']*data['dens']
                else:
                    stream1 = data[kwargs['stream'] + '1']
                    stream2 = data[kwargs['stream'] + '2']

                if kwargs['stream_average']:
                    vals_r_right = np.mean(stream1, axis=0).T
                    vals_r_left = vals_r_right
                    vals_theta_right = np.mean(stream2, axis=0).T
                    vals_theta_left = -vals_theta_right
                else:
                    vals_r_right = stream1[0, :, :].T
                    vals_r_left = stream1[int(nx3/2), :, :].T
                    vals_theta_right = stream2[0, :, :].T
                    vals_theta_left = -stream2[int(nx3/2), :, :].T

        # Join vector data through boundaries
        if kwargs['stream'] is not None:
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
        if kwargs['stream'] is not None:
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

        grid_X[idx].append(x_grid)
        grid_Y[idx].append(y_grid)
        bg[idx].append(vals)

        if kwargs['stream'] is not None:
            stream_X[idx].append(x_stream)
            vals_X[idx].append(vals_x.T)
            if kwargs['midplane']:
                stream_Y[idx].append(y_stream)
                vals_Y[idx].append(vals_y.T)
            else:
                stream_Y[idx].append(z_stream)
                vals_Y[idx].append(vals_z.T)

    # Determine colormapping properties
    cmap = plt.get_cmap(kwargs['colormap'])
    if kwargs['vmin'] is None:
        vmin = np.nanmin(bg)
    else:
        vmin = kwargs['vmin']
    if kwargs['vmax'] is None:
        vmax = np.nanmax(bg)
    else:
        vmax = kwargs['vmax']
    if kwargs['logc']:
        norm = colors.LogNorm(vmin, vmax)
    else:
        norm = colors.Normalize(vmin, vmax)

    # Make plot
    if kwargs['stream'] is not None:
        max_mag = np.nanmax(np.sqrt([[num**2 for num in lst] for lst in vals_X] + [[num**2 for num in lst] for lst in vals_Y]))


    for i in range(n):
        if n>1:
            ax = axs[0][i]
            ax_insert = axs[1][i]#fig.add_subplot(2, n, i+2)
        else:
            ax = axs[0]
            ax_insert = axs[1]#fig.add_subplot(2, n, 3)

        PCM = ax.pcolormesh(grid_X[i][0], grid_Y[i][0], bg[i][0], cmap=cmap, norm=norm)
        ax_insert.pcolormesh(grid_X[i][0], grid_Y[i][0], bg[i][0], cmap=cmap, norm=norm)

        if kwargs['streamline_width']:
            magnitude = np.sqrt(vals_X[i][0].T**2 + vals_Y[i][0].T**2)
            lw = 5*magnitude/max_mag
            lw[lw<0.15] = 0.15
            lw[lw>1.0] = 1.0
        else:
            lw = 0.5
        if kwargs['stream'] is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    'invalid value encountered in greater_equal',
                    RuntimeWarning,
                    'numpy')
                ax.streamplot(stream_X[i][0], stream_Y[i][0], vals_X[i][0], vals_Y[i][0],
                               density=kwargs['stream_density'], broken_streamlines=False,
                               linewidth=lw, arrowsize=0.3, color='w')

                U = vals_X[i][0]
                mask = np.zeros(U.shape, dtype=bool)
                mask[0:100, 0:100] = True
                mask[25:75, 25:75] = False
                #U[:50, :100] = np.nan
                U = np.ma.array(U, mask=mask)
                ax_insert.streamplot(stream_X[i][0], stream_Y[i][0], U, vals_Y[i][0],
                               density=1.5*kwargs['stream_density'], broken_streamlines=False,
                               linewidth=lw, arrowsize=1, color='w')

        ax.set_xlim((-r_max, r_max))
        ax.set_ylim((-r_max, r_max))
        ax_insert.set_xlim(-100, 100)
        ax_insert.set_ylim(-100, 100)

        # define corners to connect, clockwise from top left
        l1a, l2a = 1, 2 # start location of lines on original axis
        l1b, l2b = 4, 3 # end point of lines on zoom axis
        mark_inset(ax, ax_insert, loc1a=l1a, loc1b=l1b, loc2a=l2a, loc2b=l2b, facecolor="none", edgecolor="black")

        if kwargs['logr']:
            ax.set_xlabel(r'$\log_{10}(r)\ x / r$')
            ax_insert.set_xlabel(r'$\log_{10}(r)\ x / r$')
            if kwargs['midplane']:
                ax.set_ylabel(r'$\log_{10}(r)\ y / r$')
                ax_insert.set_ylabel(r'$\log_{10}(r)\ y / r$')
            else:
                ax.set_ylabel(r'$\log_{10}(r)\ z / r$')
                ax_insert.set_ylabel(r'$\log_{10}(r)\ z / r$')
        else:
            ax.set_xlabel(r'$x$')
            ax_insert.set_xlabel(r'$x$')
            if kwargs['midplane']:
                ax.set_ylabel(r'$y$')
                ax_insert.set_ylabel(r'$y$')
            else:
                ax.set_ylabel(r'$z$')
                ax_insert.set_ylabel(r'$z$')
        if kwargs['time']:
            ax.set_title("Time: {:.2e}".format(titles[i]))
        if kwargs['minmax']:
            min_all = np.min(scalar_q)
            max_all = np.max(scalar_q)
            txt = 'Min: %f Max: %f' % (min_all,max_all)
            plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        ax.set_aspect('equal')
        ax_insert.set_aspect('equal')

        if n>1:
            for ax_f in axs.flat:
                ax_f.label_outer()

        at = AnchoredText(labels[i], prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

    # make colorbar
    if n>1:
        ax = axs[0][n-1]
    else:
        ax = axs[0][n-1]
    cax = ax.inset_axes([1.04, 0, 0.05, 1.0])
    cbar = fig.colorbar(PCM, ax=ax, cax=cax)
    cbar.ax.set_title(cbar_title)

    if kwargs['output_file'] == 'show':
        plt.show()
    else:
        plt.savefig(kwargs['output_file'], bbox_inches='tight', dpi=kwargs['dpi'])

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
# loc1, loc2 : {1, 2, 3, 4}
def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, zorder=10, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',
                        nargs='+',
                        help='name of athdf file to be plotted, possibly including path')
    parser.add_argument('-q',
                        '--quantity',
                        help='name of quantity to be plotted in background')
    parser.add_argument('-o',
                        '--output_file',
                        help=('name of output to be (over)written, possibly including '
                              'path; use "show" to show interactive plot instead'))
    parser.add_argument('-m',
                        '--midplane',
                        action='store_true',
                        help=('flag indicating plot should be midplane (r,phi) rather '
                              'than (r,theta)'))
    parser.add_argument('-a', '--average',
                        action='store_true',
                        help='flag indicating phi-averaging should be done')
    parser.add_argument('-r', '--r_max',
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
    parser.add_argument('--minmax',
                        action='store_true',
                        help=('flag indicating global max and min should be plotted'))
    parser.add_argument('--logc',
                        action='store_true',
                        help='flag indicating data should be colormapped logarithmically')
    parser.add_argument('-s', '--stream',
                        default=None,
                        help='name of vector quantity to use to make stream plot')
    parser.add_argument('--stream_average',
                        action='store_true',
                        help='flag indicating phi-averaging on stream plot data')
    parser.add_argument('--streamline_width',
                        action='store_true',
                        help='flag indicating streamlines should vary in width according to magnitude')
    parser.add_argument('--stream_density',
                        type=float,
                        default=1.0,
                        help='density of stream lines')
    parser.add_argument('--stream_samples',
                        type=int,
                        default=100,
                        help='linear size of stream line sampling grid')
    parser.add_argument('-t',
                        '--time',
                        action='store_true',
                        help=('flag indicating title should be set to time'))
    parser.add_argument('--dpi',
                        type=int,
                        default=250,
                        help='resolution of saved image (dots per square inch, default: 250)')
    args = parser.parse_args()
    main(**vars(args))
