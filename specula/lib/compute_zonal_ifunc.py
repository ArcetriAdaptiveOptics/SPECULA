from scipy.interpolate import Rbf
import numpy as np
from specula.lib.make_mask import make_mask
from specula import cpuArray

def compute_zonal_ifunc(dim, n_act, xp=np, dtype=np.float32,circ_geom:bool=False,
                        geom:str=None, angle_offset=0,
                        do_mech_coupling=False, coupling_coeffs=[0.31, 0.05],
                        do_slaving=False, slaving_thr=0.1,
                        obsratio=0.0, diaratio=1.0, mask=None, return_coordinates=False):
    """ Computes the ifs_cube matrix with Influence Functions using Thin Plate Splines """

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True, xp=xp)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)

    step = float(dim) / float(n_act)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    if circ_geom is True:
        if geom is not None:
            raise ValueError(f'Too many geometry inputs! Both circ_geom = {circ_geom} and geom = {geom} were given')
        geom = 'circular' # added for retro-compatibility
    else:
        if geom is None:
            geom = 'square' # default geometry
                          
    # Actuator Coordinates
    if geom == 'circular':
        if n_act % 2 == 0:
            na = xp.arange(xp.ceil((n_act + 1) / 2)) * 6
        else:
            step *= float(n_act) / float(n_act - 1)
            na = xp.arange(xp.ceil(n_act / 2)) * 6
        na[0] = 1  # The first value is always 1
        n_act_tot = int(xp.sum(na))
        pol_coords = xp.zeros((2, n_act_tot))
        ka = 0
        # Refactor this!
        for ia in range(len(na)):
            n_angles = int(na[ia])
            for ja in range(n_angles):
                pol_coords[0, ka] = 360. / na[ia] * ja + angle_offset  # Angle in degrees
                pol_coords[1, ka] = ia * step  # Radial distance
                ka += 1
        # System center
        x_c, y_c = dim / 2, dim / 2
        # Convert from polar to Cartesian coordinates
        x = pol_coords[1] * xp.cos(xp.radians(pol_coords[0])) + x_c
        y = pol_coords[1] * xp.sin(xp.radians(pol_coords[0])) + y_c
    
    elif geom == 'alpao':
        x, y = xp.meshgrid(xp.linspace(0, dim, n_act), xp.linspace(0, dim, n_act))
        x, y = x.ravel(), y.ravel()
        x_c, y_c = dim / 2, dim / 2 # center
        rho = xp.sqrt((x-x_c)**2+(y-y_c)**2)
        rho_max = (dim*(9/8-n_act/(24*16)))/2 # slightly larger than dim, depends on n_act
        x = x[rho<=rho_max]
        y = y[rho<=rho_max]
        n_act_tot = int(xp.size(x))
      
    elif geom == 'square': # default
        x, y = xp.meshgrid(xp.linspace(0, dim, n_act), xp.linspace(0, dim, n_act))
        x, y = x.ravel(), y.ravel()
        n_act_tot = n_act**2
      
    else:
      raise ValueError("Unrecognized geometry type! Avaliable types are: 'circular', 'alpao', 'square'")

    coordinates = xp.vstack((x, y))
    grid_x, grid_y = xp.meshgrid(xp.arange(dim), xp.arange(dim))

    # ----------------------------------------------------------
    # Influence Function (ifs_cube) Computation
    ifs_cube = xp.zeros((n_act_tot, dim, dim), dtype=dtype)

    # Minimum distance between points
    min_distance_norm = 9*dim/n_act

    for i in range(n_act_tot):
        z = xp.zeros(n_act_tot, dtype=dtype)
        z[i] = 1.0  # Set the central actuator

        if min_distance_norm >= dim/2:
            x_close, y_close, z_close = x, y, z
            idx_far_grid = None
        else:
            distance = xp.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
            idx_close = xp.where(distance <= min_distance_norm)[0]
            x_close, y_close, z_close = x[idx_close], y[idx_close], z[idx_close]           
            # Compute the distance grid
            distance_grid = xp.sqrt((grid_x.ravel() - x[i]) ** 2 + (grid_y.ravel() - y[i]) ** 2)
            idx_far_grid = xp.where(distance_grid > 0.8*min_distance_norm)[0]

        # Convert to NumPy arrays for Rbf interpolation (required)
        x_close_np = cpuArray(x_close)
        y_close_np = cpuArray(y_close)
        z_close_np = cpuArray(z_close)
        grid_x_np = cpuArray(grid_x)
        grid_y_np = cpuArray(grid_y)

        # Interpolation using Thin Plate Splines (using NumPy arrays)
        rbf = Rbf(x_close_np, y_close_np, z_close_np, function='thin_plate')

        # Perform interpolation
        z_interp_np = rbf(grid_x_np, grid_y_np)

        # Convert back to xp array
        z_interp = xp.asarray(z_interp_np)

        if idx_far_grid is not None:
            z_interp.ravel()[idx_far_grid] = 0

        ifs_cube[i, :, :] = z_interp

        print(f"\rCompute IFs: {int((i / n_act_tot) * 100)}% done", end="")

    print()

    if do_mech_coupling:
        print("Applying mechanical coupling...")
        ifs_cube_orig = ifs_cube.copy()

        for j in range(n_act_tot):
            # Distance from actuator j to all others
            distance = xp.sqrt((x - x[j])**2 + (y - y[j])**2)

            # Find neighbors, excluding self (distance > 0)
            close1_indices = xp.where((distance > 0) & (distance <= step))[0]
            close2_indices = xp.where((distance > step) & (distance <= 2 * step))[0]

            # Start with original influence function
            ifs_cube[j, :, :] = ifs_cube_orig[j, :, :]

            # Add coupling contributions
            if len(close1_indices) > 0:
                ifs_cube[j, :, :] += coupling_coeffs[0] * xp.sum(ifs_cube_orig[close1_indices], axis=0)

            if len(close2_indices) > 0:
                ifs_cube[j, :, :] += coupling_coeffs[1] * xp.sum(ifs_cube_orig[close2_indices], axis=0)

        print("Mechanical coupling applied.")

    if do_slaving:
        max_vals = xp.max(ifs_cube[:, idx[0], idx[1]], axis=1)
        max_vals_all = xp.max(ifs_cube, axis=(1, 2))
        idx_master = xp.where(max_vals >= slaving_thr * max_vals_all)[0]
        idx_slave = xp.where(max_vals < slaving_thr * max_vals_all)[0]

        print(f"Actuators: {n_act_tot}")
        print(f"Master actuators: {len(idx_master)}")
        print(f"Actuators to be slaved: {len(idx_slave)}")

        slaveMat1 = xp.zeros((n_act_tot, n_act_tot), dtype=dtype)

        for i in range(n_act_tot):
            if i in idx_master:
                distance = xp.sqrt((coordinates[0] - coordinates[0][i])**2 + 
                                (coordinates[1] - coordinates[1][i])**2)

                idx_close_master1 = xp.where(distance <= 1.1 * step)[0]
                idx_close_master1 = xp.intersect1d(idx_close_master1, idx_slave)

                if len(idx_close_master1) > 0:
                    for j in idx_close_master1:
                        slaveMat1[i, j] = 1.0

        for j in range(n_act_tot):
            slaveMat1[:, j] *= 1.0 / max(1.0, xp.sum(slaveMat1[:, j]))

        for i in range(n_act_tot):
            if xp.sum(slaveMat1[i, :]) > 0:
                idx_temp = xp.where(slaveMat1[i, :] > 0)[0]
                for j in idx_temp:
                    ifs_cube[i] += slaveMat1[i, j] * ifs_cube[j]

        ifs_cube = ifs_cube[idx_master]
        coords = coordinates[:, idx_master]
        n_act_tot = len(idx_master)

    ifs_2d = xp.array([ifs_cube[i][idx] for i in range(n_act_tot)], dtype=dtype)

    print("\nComputation completed.")

    if return_coordinates:
        return ifs_2d, mask, coords
    else:
        return ifs_2d, mask
