# Python standard modules
import argparse
import sys
import os

dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, '..', '..', 'dependencies')
sys.path.append(lib_path)

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read


# Main function
def main(**kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Set resolution of plot in dots per square inch
    if kwargs['dpi'] is not None:
        resolution = kwargs['dpi']

    data = athena_read.athdf(kwargs['data_file'])

    if kwargs['background'] not in [list(data.keys()),'dvp','vor','vorb']:
        print('Background choice invalid. Choose from data keys.')
    if kwargs['contour'] is not None and kwargs['contour'] not in [list(data.keys()),'dvp','vor','vorb']:
        print('Contour choice invalid. Choose from data keys.')

    # Extract basic coordinate information
    r = data['x1v']
    theta = data['x2v']
    phi = data['x3v']
    r_face = data['x1f']
    theta_face = data['x2f']
    phi_face = data['x3f']
    nx2 = len(theta)

    phi_grid, theta_grid, r_grid = np.meshgrid(phi_face,theta_face,r_face,indexing='ij') # req all comp for curl later

    # Set radial extent
    if kwargs['r_max'] is not None:
        r_max = kwargs['r_max']
    else:
        r_max = r_face[-1]

    if kwargs['background'] in ['dvp','vor','vorb'] or kwargs['contour'] in ['dvp','vor','vorb']:
        # Calculate velocities
        vr = data['mom1']/data['dens']
        vt = data['mom1']/data['dens']
        vp = data['mom1']/data['dens']

        # Subtract background angular velocity
        r_mesh,_,_ = np.meshgrid(phi,theta,r, sparse=False, indexing='ij')
        GM = 1.
        Omega_kep = np.sqrt(GM/(r**3.))
        Omega_kep = np.broadcast_to(Omega_kep, (np.shape(vp)[0], np.shape(vp)[1], np.shape(vp)[2]))
        dvp = vp - r_mesh*Omega_kep

        if kwargs['background'] in ['vor','vorb'] or kwargs['contour'] in ['vor','vorb']:
            # Define the vortensity
            _, _, w3 = curl(r_grid,theta_grid,phi_grid,vr,vt,dvp)
            if kwargs['background'] in ['vor'] or kwargs['contour'] in ['vor']:
                vor3 = w3/data['dens']
            if kwargs['background'] in ['vorb'] or kwargs['contour'] in ['vorb']:
                vorb3 = w3*data['dens']/(data['Bcc1']**2. + data['Bcc2']**2. + data['Bcc3']**2.)

    if kwargs['background']=='dvp':
        background = dvp
    elif kwargs['background']=='vor':
        background = vor3
    elif kwargs['background']=='vorb':
        background = vorb3
    else:
        background = data[kwargs['background']]

    if kwargs['contour'] is not None:
        if kwargs['contour']=='dvp':
            contour = dvp
        elif kwargs['contour']=='vor':
            contour = vor3
        elif kwargs['contour']=='vorb':
            contour = vorb3
        else:
            contour = data[kwargs['contour']]


    # Perform slicing/averaging of background
    if nx2 % 2 == 0:
        bg_vals = np.mean(background[:, int(nx2/2-1):int(nx2/2+1), :], axis=1)
        if kwargs['contour'] is not None:
            ct_vals = np.mean(contour[:, int(nx2/2-1):int(nx2/2+1), :], axis=1)
    else:
        bg_vals = background[:, int(nx2/2), :]
        if kwargs['contour'] is not None:
            ct_vals = contour[:, int(nx2/2), :]

    # Create background scalar grid
    phi_grid_alt, _, r_grid_alt = np.meshgrid(phi, theta, r, indexing='ij')
    # make 2D plotting grid at midplane
    x_grid = r_grid[:, int(nx2/2), :] * np.cos(phi_grid[:, int(nx2/2), :])
    y_grid = r_grid[:, int(nx2/2), :] * np.sin(phi_grid[:, int(nx2/2), :])
    x_grid_alt = r_grid_alt[:, int(nx2/2), :] * np.cos(phi_grid_alt[:, int(nx2/2), :])
    y_grid_alt = r_grid_alt[:, int(nx2/2), :] * np.sin(phi_grid_alt[:, int(nx2/2), :])

    _, ax = plt.subplots()
    if kwargs['background'] in ['dvp','vor','vorb']:
        val_max = np.max(np.abs(bg_vals))
        im = ax.pcolormesh(x_grid, y_grid, bg_vals, norm=colors.SymLogNorm(linthresh=0.001*val_max,linscale=0.5,base=10), cmap='seismic')
    else:
        im = ax.pcolormesh(x_grid, y_grid, bg_vals, cmap='seismic')
    plt.colorbar(im)
    if kwargs['contour'] is not None:
        CS = ax.contour(x_grid_alt, y_grid_alt, ct_vals, colors='k')
        ax.clabel(CS, inline=True, fontsize=10)

    plt.gca().set_aspect('equal')
    plt.xlim((-r_max, r_max))
    plt.ylim((-r_max, r_max))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',
                        help='name of athdf file, possibly including path')
    parser.add_argument('background',
                        help='name of quantity to be plotted as colormap')
    parser.add_argument('output_file',
                        help=('name of output to be (over)written, possibly including '
                              'path; use "show" to show interactive plot instead'))
    parser.add_argument('--contour',
                        type=str,
                        default=None,
                        help='name of quantity to be plotted as contours (optional)')
    parser.add_argument('-r',
                        '--r_max',
                        type=float,
                        default=None,
                        help='maximum radial extent of plot')
    parser.add_argument('--dpi',
                        type=int,
                        default=None,
                        help='resolution of saved image (dots per square inch)')
    args = parser.parse_args()
    main(**vars(args))
