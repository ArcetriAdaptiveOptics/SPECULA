from scipy.interpolate import Rbf
import numpy as np
from specula.lib.make_mask import make_mask
from specula import cpuArray


def compute_zonal_ifunc(dim, n_act, xp=np, dtype=np.float32, circ_geom:bool=False,
                        geom:str=None, angle_offset=0,
                        do_mech_coupling=False, coupling_coeffs=[0.31, 0.05],
                        do_slaving=False, slaving_thr=0.1, linear_slaving=False,
                        edge_constraint_weight=0.0, search_radius_steps=2.5,
                        obsratio=0.0, diaratio=1.0, mask=None):
    """
    Computes the ifs_cube matrix with Influence Functions using Thin Plate Splines
    
    Parameters
    ----------
    
    dim : int
        Size of the output influence function maps (dim x dim).
    n_act : int
        Number of actuators across the diameter (for circular) or side (for square).
    xp : module
        Array module (numpy or cupy).
    dtype : dtype
        Data type for the output influence functions.
    circ_geom : bool
        If True, use circular geometry with actuator counts per ring defined by
        na = [1, 6, 12, 18, ...].
    geom : str or None
        Geometry type: 'circular', 'alpao', or 'square'. If None, defaults to 'square' unless
        circ_geom is True.
    angle_offset : float
        Angular offset in degrees for the circular geometry actuator placement.
    do_mech_coupling : bool
        If True, apply mechanical coupling to the influence functions.
    coupling_coeffs : list of two floats
        Coupling coefficients for first and second nearest neighbors when do_mech_coupling is True.
    do_slaving : bool
        If True, apply actuator slaving to reduce the number of independent actuators.
    slaving_thr : float
        Threshold (as a fraction of peak IF) to classify master vs slave actuators for slaving.
    linear_slaving : bool
        If True, use linear (piston + tip + tilt) extrapolation for slaving instead of simple
        weighted averaging.
    edge_constraint_weight : float
        Weight of a virtual zero-valued actuator placed one step outward from each slaved actuator
        along the radial direction from the array center, used only for linear_slaving.
    search_radius_steps : float
        Radius (in units of step) to search for nearby masters for the plane fit in linear_slaving.
        Falls back to all masters if fewer than 3 are found within this radius.
    obsratio : float
        Ratio of obstructed to total area in the pupil mask (0 to 1). Used to generate the mask if
        mask is None.
    diaratio : float
        Ratio of inner to outer diameter in the pupil mask (0 to 1). Used to generate the mask if
        mask is None.
    mask : array or None
        Optional pre-computed pupil mask. If None, it will be generated using obsratio and diaratio.
    """

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True, xp=xp)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    if circ_geom is True:
        if geom is not None:
            raise ValueError(f'Too many geometry inputs! Both circ_geom'
                             f' = {circ_geom} and geom = {geom} were given')
        geom = 'circular' # added for retro-compatibility
    else:
        if geom is None:
            geom = 'square' # default geometry

    # Actuator Coordinates
    if geom == 'circular':
        if n_act % 2 == 0:
            n_act_radius = int(xp.ceil((n_act + 1) / 2))
        else:
            n_act_radius = int(xp.ceil(n_act / 2))
        na = xp.arange(n_act_radius) * 6
        na[0] = 1  # The first value is always 1
        n_act_tot = int(xp.sum(na))

        # Calculate step based on number of actuators on diameter
        n_act_diameter = 2 * n_act_radius - 1
        step = float(dim - 1) / float(n_act_diameter - 1)

        pol_coords = xp.zeros((2, n_act_tot))
        ka = 0

        # Refactor this!
        for ia, _ in enumerate(na):
            n_angles = int(na[ia])
            for ja in range(n_angles):
                pol_coords[0, ka] = 360. / na[ia] * ja + angle_offset  # Angle in degrees
                pol_coords[1, ka] = ia * step  # Radial distance
                ka += 1

        # System center - use (dim-1)/2 to properly center on the grid
        x_c, y_c = (dim - 1) / 2.0, (dim - 1) / 2.0

        # Convert from polar to Cartesian coordinates
        x = pol_coords[1] * xp.cos(xp.radians(pol_coords[0])) + x_c
        y = pol_coords[1] * xp.sin(xp.radians(pol_coords[0])) + y_c

    elif geom == 'alpao':
        x, y = xp.meshgrid(xp.linspace(0, dim - 1, n_act), xp.linspace(0, dim - 1, n_act))
        x, y = x.ravel(), y.ravel()
        x_c, y_c = (dim - 1) / 2.0, (dim - 1) / 2.0 # center
        rho = xp.sqrt((x-x_c)**2+(y-y_c)**2)
        rho_max = ((dim - 1)*(9/8-n_act/(24*16)))/2 # slightly larger than (dim-1)/2, depends on n_act
        x = x[rho<=rho_max]
        y = y[rho<=rho_max]
        n_act_tot = int(xp.size(x))
        # Calculate step based on linspace spacing
        step = float(dim - 1) / float(n_act - 1)

    elif geom == 'square': # default
        x, y = xp.meshgrid(xp.linspace(0, dim - 1, n_act), xp.linspace(0, dim - 1, n_act))
        x, y = x.ravel(), y.ravel()
        n_act_tot = n_act**2
        # Calculate step based on linspace spacing
        step = float(dim - 1) / float(n_act - 1)

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
                ifs_cube[j, :, :] += coupling_coeffs[0] * \
                    xp.sum(ifs_cube_orig[close1_indices], axis=0)

            if len(close2_indices) > 0:
                ifs_cube[j, :, :] += coupling_coeffs[1] * \
                    xp.sum(ifs_cube_orig[close2_indices], axis=0)

        print("Mechanical coupling applied.")

    if do_slaving:
        ifs_cube, coordinates, n_act_tot, slave_mat = apply_slaving(
            ifs_cube=ifs_cube,
            coordinates=coordinates,
            idx=idx,
            step=step,
            slaving_thr=slaving_thr,
            linear=linear_slaving,
            edge_constraint_weight=edge_constraint_weight,
            search_radius_steps=search_radius_steps,
            xp=xp,
            dtype=dtype
        )
        coords = coordinates
    else:
        coords = coordinates

    ifs_2d = xp.array([ifs_cube[i][idx] for i in range(n_act_tot)], dtype=dtype)

    print("\nComputation completed.")

    # If do_slaving=False, slave_mat might not be defined. Handle safely:
    if not do_slaving:
        slave_mat = xp.zeros((n_act_tot, n_act_tot), dtype=dtype)
    return ifs_2d, mask, coords, slave_mat


# ==============================================================================
# SLAVING HELPER FUNCTIONS
# ==============================================================================

def _compute_weights_standard(coordinates, idx_master, idx_slave, step, xp, dtype):
    """
    Computes the weight matrix W_ms (Master x Slave) using the standard proximity method.
    """
    n_masters = len(idx_master)
    n_slaves = len(idx_slave)
    W_ms = xp.zeros((n_masters, n_slaves), dtype=dtype)

    # For each master, find the nearby slaves
    for i, m_idx in enumerate(idx_master):
        dist_to_slaves = xp.sqrt(
            (coordinates[0, idx_slave] - coordinates[0, m_idx])**2 +
            (coordinates[1, idx_slave] - coordinates[1, m_idx])**2
        )
        close_mask = dist_to_slaves <= 1.1 * step
        W_ms[i, close_mask] = 1.0

    # Normalize columns so that the sum of weights for each slave is 1 (if > 0)
    col_sums = xp.sum(W_ms, axis=0)
    col_sums = xp.where(col_sums > 1.0, col_sums, 1.0)
    W_ms /= col_sums

    return W_ms

def _compute_weights_linear(coordinates, idx_master, idx_slave, step,
                            edge_constraint_weight, search_radius_steps, xp, dtype):
    """
    Computes the weight matrix W_ms using linear extrapolation (Piston, Tip, Tilt).
    """

    coords_np = cpuArray(coordinates).astype(float)
    idx_master_np = cpuArray(idx_master)
    idx_slave_np = cpuArray(idx_slave)

    x_c = np.mean(coords_np[0])
    y_c = np.mean(coords_np[1])
    search_radius = search_radius_steps * step

    W_ms_np = np.zeros((len(idx_master_np), len(idx_slave_np)), dtype=float)

    for j_idx, j in enumerate(idx_slave_np):
        xj = coords_np[0, j]
        yj = coords_np[1, j]

        dist_to_masters = np.sqrt(
            (coords_np[0, idx_master_np] - xj)**2 +
            (coords_np[1, idx_master_np] - yj)**2
        )

        mask_nearby = dist_to_masters <= search_radius
        nearby_indices = np.where(mask_nearby)[0]

        # Fallback: if fewer than 3 masters are within the search radius, take the 3 closest
        if len(nearby_indices) < 3:
            nearby_indices = np.argsort(dist_to_masters)[:3]

        nearby_masters = idx_master_np[nearby_indices]
        xm = coords_np[0, nearby_masters]
        ym = coords_np[1, nearby_masters]

        A = np.column_stack([xm, ym, np.ones(len(xm))])
        pj = np.array([xj, yj, 1.0])

        if edge_constraint_weight > 0.0:
            dx = xj - x_c
            dy = yj - y_c
            dist_from_center = np.sqrt(dx**2 + dy**2)
            if dist_from_center > 1e-10:
                dx_n, dy_n = dx / dist_from_center, dy / dist_from_center
            else:
                dx_n, dy_n = 1.0, 0.0

            # Virtual actuator placed one step outward
            xv = xj + step * dx_n
            yv = yj + step * dy_n

            A_aug = np.vstack([A, edge_constraint_weight * np.array([[xv, yv, 1.0]])])
            ATA = A_aug.T @ A_aug
        else:
            ATA = A.T @ A

        try:
            v = np.linalg.solve(ATA, pj)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(ATA, pj, rcond=None)[0]

        w_j = A @ v
        W_ms_np[nearby_indices, j_idx] = w_j

    return xp.asarray(W_ms_np, dtype=dtype)


def apply_slaving(ifs_cube, coordinates, idx, step, slaving_thr=0.1,
                  linear=False, edge_constraint_weight=0.0, search_radius_steps=2.5,
                  xp=np, dtype=np.float32):
    """
    Unified function to apply actuator slaving. 
    Routes to standard (proximity) or linear (PTT extrapolation) weighting logic.
    """
    n_act_tot = ifs_cube.shape[0]

    # --- 1. PRE-PROCESSING (Identify Master/Slave actuators) ---
    ifs_peaks = xp.max(ifs_cube[:, idx[0], idx[1]], axis=1)
    max_vals_all = xp.max(ifs_cube, axis=(1, 2))

    idx_master = xp.where(ifs_peaks >= slaving_thr * max_vals_all)[0]
    idx_slave = xp.where(ifs_peaks < slaving_thr * max_vals_all)[0]

    print(f"Actuators: {n_act_tot}")
    print(f"Master actuators: {len(idx_master)}")
    print(f"Actuators to be slaved: {len(idx_slave)}")

    slave_mat = xp.zeros((n_act_tot, n_act_tot), dtype=dtype)

    if len(idx_slave) == 0:
        return ifs_cube[idx_master], coordinates[:, idx_master], len(idx_master), slave_mat

    # --- 2. WEIGHT COMPUTATION (Routing to Standard or Linear logic) ---
    if linear:
        W_ms = _compute_weights_linear(coordinates, idx_master, idx_slave, step, 
                                       edge_constraint_weight, search_radius_steps, xp, dtype)
    else:
        W_ms = _compute_weights_standard(coordinates, idx_master, idx_slave, step, xp, dtype)

    # --- 3. POST-PROCESSING (Vectorized application and output) ---
    ifs_masters = ifs_cube[idx_master] 
    ifs_slaves = ifs_cube[idx_slave]   

    # Tensordot performs: sum_j ( W_ms[i, j] * ifs_slaves[j, y, x] )
    slave_contributions = xp.tensordot(W_ms, ifs_slaves, axes=([1], [0]))

    ifs_cube_out = ifs_masters + slave_contributions
    coords_out = coordinates[:, idx_master]

    # Update the slave_mat using xp.ix_ for correct numpy/cupy indexing
    slave_mat[xp.ix_(idx_master, idx_slave)] = W_ms

    return ifs_cube_out, coords_out, len(idx_master), slave_mat
