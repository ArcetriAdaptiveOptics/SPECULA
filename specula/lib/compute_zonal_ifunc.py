from scipy.interpolate import Rbf
from specula.lib.make_mask import make_mask

def compute_zonal_ifunc(dim, n_act, xp, dtype, circ_geom=False, angle_offset=0,
                        do_mech_coupling=False, coupling_coeffs=[0.31, 0.05],
                        obsratio=0.0, diaratio=1.0, mask=None):
    """ Computes the ifs_cube matrix with Influence Functions using Thin Plate Splines """

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True, xp=xp)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)[0]

    step = float(dim) / float(n_act)

    # ----------------------------------------------------------
    # Actuator Coordinates
    if circ_geom:
        if n_act % 2 == 0:
            step *= float(n_act) / float(n_act - 1)
            na = xp.arange(round((n_act + 1) / 2)) * 6
        else:
            na = xp.arange(round(n_act / 2)) * 6
        na[0] = 1  # The first value is always 1

        n_act_tot = xp.sum(na)
        pol_coords = xp.zeros((2, n_act_tot))
        ka = 0
        for ia in range(len(na)):
            for ja in range(na[ia]):
                pol_coords[0, ka] = 360. / na[ia] * ja + angle_offset  # Angle in degrees
                pol_coords[1, ka] = ia * step  # Radial distance
                ka += 1

        # System center
        x_c, y_c = dim / 2, dim / 2  

        # Convert from polar to Cartesian coordinates
        x, y = pol_coords[1] * xp.cos(xp.radians(pol_coords[0])), pol_coords[1] * xp.sin(xp.radians(pol_coords[0]))
        x += x_c  # Shift to the center of the grid
        y += y_c  

        # Maximum radius (outer boundary)
        R = pol_coords[1].max()  # The maximum radial value is the outer boundary
    else:
        x, y = xp.meshgrid(xp.linspace(0, dim, n_act), xp.linspace(0, dim, n_act))
        x, y = x.ravel(), y.ravel()
        n_act_tot = n_act ** 2

    coordinates = xp.vstack((x, y))
    grid_x, grid_y = xp.meshgrid(xp.arange(dim), xp.arange(dim))

    # ----------------------------------------------------------
    # Influence Function (ifs_cube) Computation
    ifs_cube = xp.zeros((n_act_tot, dim, dim))

    # Minimum distance between points
    min_distance_norm = 9*dim/n_act

    for i in range(n_act_tot):
        z = xp.zeros(n_act_tot)

        z[i] = 1.0  # Set the central actuator
        
        if min_distance_norm >= dim/2:
            x_close, y_close, z_close = x, y, z
            idx_far_grid = None
        else:
            distance = xp.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            idx_close = xp.where(distance <= min_distance_norm)[0]
            x_close, y_close, z_close = x[idx_close], y[idx_close], z[idx_close]           
            # Compute the distance grid
            distance_grid = xp.sqrt((grid_x.flat - x[i]) ** 2 + (grid_x.flat - y[i]) ** 2)
            idx_far_grid = xp.where(distance_grid > 0.8*min_distance_norm)[0]

        # Interpolation using Thin Plate Splines
        rbf = Rbf(x_close, y_close, z_close, function='thin_plate')
        
        z_interp = rbf(grid_x, grid_y)
        if idx_far_grid is not None:
            z_interp.flat[idx_far_grid] = 0

        ifs_cube[i, :, :] = z_interp

        # Mechanical Coupling
        if do_mech_coupling:
            ifs_cube_orig = ifs_cube.copy()
            for j in range(n_act_tot):
                distance = xp.sqrt((x - x[j])**2 + (y - y[j])**2)

                close1_set = xp.where(distance <= step)[0]
                close2_set = xp.where((distance > step) & (distance <= 2 * step))[0]

                ifs_cube[j, :, :] = ifs_cube_orig[j, :, :]

                if len(close1_set) > 0:
                    for k in close1_set:
                        ifs_cube[j, :, :] += coupling_coeffs[0] * ifs_cube_orig[k, :, :]

                if len(close2_set) > 0:
                    for k in close2_set:
                        ifs_cube[j, :, :] += coupling_coeffs[1] * ifs_cube_orig[k, :, :]

        print(f"\rCompute IFs: {int((i / n_act_tot) * 100)}% done", end="")

    ifs_2d = xp.array([ifs_cube[i][idx] for i in range(n_act_tot)], dtype=dtype)

    print("\nComputation completed.")

    return ifs_2d, mask