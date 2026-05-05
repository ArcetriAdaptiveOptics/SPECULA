import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C

def display_ifunc_2d(ifunc_obj: IFunc, m2c_obj: M2C = None, modal_vector=None,
                     id_mode_starting: int = 0, n_raw_col: int = 10,
                     do_not_show_ticks: bool = False, show_plot: bool = True):
    """
    Displays an influence function or reconstructed modes from an IFunc object.
    
    Args:
        ifunc_obj (IFunc): The Influence Function object containing the modal base and mask.
        m2c_obj (M2C, optional): Modal to Command data object used to multiply the base.
        modal_vector (array-like, optional): Vector(s) of modal coefficients to reconstruct shapes.
        id_mode_starting (int, optional): The ID of the first mode to display. Defaults to 0.
        n_raw_col (int, optional): Number of rows/cols for the grid display. Defaults to 10.
        do_not_show_ticks (bool, optional): If True, hides the axes ticks. Defaults to False.
        show_plot (bool, optional): If True, displays the plot. Set to False for testing.
                                    Defaults to True.

    Returns:
        numpy.ndarray: The computed 2D shape or grid of shapes.
    """

    # --- 1. Extract and format data from the IFunc object ---
    # We use cpuArray to safely bring the data to CPU memory as a numpy array,
    # which is required by matplotlib.
    modal_base = cpuArray(ifunc_obj.influence_function).astype(float)
    mask_small = cpuArray(ifunc_obj.mask_inf_func)

    # Apply Modal to Command matrix if provided
    if m2c_obj is not None:
        if not hasattr(m2c_obj, 'm2c'):
            raise TypeError('m2c_obj must be an M2C object exposing the m2c matrix field.')
        m2c = np.array(cpuArray(m2c_obj.m2c), dtype=float)
        modal_base = m2c @ modal_base

    # Create a boolean mask for faster array indexing in Python
    mask_small_bool = mask_small > 0
    mask_small_size = mask_small.shape[0]

    # --- 2. Inner helper function for plotting ---
    def show_img(data, title):
        plt.figure(figsize=(12, 9))
        max_abs = np.max(np.abs(data))

        # We transpose the data (data.T) for matplotlib (Y, X) to match IDL's [X, Y] logic.
        # origin='lower' ensures the origin is at the bottom-left, just like in IDL.
        plt.imshow(data.T, origin='lower', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        plt.title(title, fontsize=14)
        plt.colorbar()

        if do_not_show_ticks:
            plt.axis('off') # Hides axes ticks

        if show_plot:
            plt.show()

    shape_out = None

    # --- 3. Visualization Logic ---

    # CASE A: A modal vector is provided, we reconstruct the shape
    if modal_vector is not None:
        modal_vector = np.array(modal_vector, dtype=float)

        # 1D Vector: Single frame reconstruction
        if modal_vector.ndim == 1:
            shape = np.zeros((mask_small_size, mask_small_size), dtype=float)
            n_elements = len(modal_vector)
            mb_subset = modal_base[0:n_elements, :]

            # Vector-Matrix multiplication (replaces IDL's vecmat_multiply)
            shape[mask_small_bool] = modal_vector @ mb_subset

            show_img(shape, 'Reconstructed Shape')
            shape_out = shape

        # 2D Vector: Multiple frames reconstruction (e.g., num_frames x num_modes)
        else:
            num_frames = modal_vector.shape[0]
            num_modes = modal_vector.shape[1]

            shape = np.zeros((mask_small_size, mask_small_size, num_frames), dtype=float)
            mb_subset = modal_base[0:num_modes, :]

            for i in range(num_frames):
                temp = np.zeros((mask_small_size, mask_small_size), dtype=float)
                temp[mask_small_bool] = modal_vector[i, :] @ mb_subset
                shape[:, :, i] = temp

            # Show the very last frame computed (-1 index in Python)
            show_img(shape[:, :, -1], 'Reconstructed Shape (Last Frame)')
            shape_out = shape

    # CASE B: No modal vector provided, display a grid of influence functions
    else:
        id_mode_starting = int(id_mode_starting)
        max_mode = modal_base.shape[0]

        # Initialize a large array for the mosaic grid
        shape_big = np.zeros((n_raw_col * mask_small_size, n_raw_col * mask_small_size),
                             dtype=float)

        for i in range(n_raw_col):
            for j in range(n_raw_col):
                # Calculate the current mode index based on grid position
                id_mode = id_mode_starting + i + (n_raw_col - 1 - j) * n_raw_col

                if id_mode < max_mode:
                    shape = np.zeros((mask_small_size, mask_small_size), dtype=float)
                    shape[mask_small_bool] = modal_base[id_mode, :]

                    # Map the small shape into the correct position in the big grid array
                    # Logic is kept identical to IDL since we maintained the [X, Y] convention
                    shape_big[i * mask_small_size : (i + 1) * mask_small_size,
                              j * mask_small_size : (j + 1) * mask_small_size] = shape

        title_str = f"Mode {id_mode_starting} - {id_mode_starting + n_raw_col**2 - 1}"
        show_img(shape_big, title_str)
        shape_out = shape_big

    return shape_out


def display_mcao_geom(diam, no_gs, gs_height, dm_height, gs_fov_diam_asec,
                      shifts=None, rotations=None, no_subaps=None,
                      tech_fov_diam_asec=None, ngs_fov_diam_asec=None,
                      sci_fov_diam_asec=None, sci_square=False, title=None,
                      figsize=None, ax=None, display_sa_lines=False,
                      gs_uniform_color=False, gs_color='gold', gs_alpha=0.5,
                      gs_circles_filled=False,
                      show_plot=True, verbose=False):
    """
    Display sub-aperture centers for an MCAO geometry.

    Args:
        diam (float): Pupil diameter.
        no_gs (int): Number of Guide Stars.
        gs_height (float): Guide Star height [m].
        dm_height (float): DM height [m].
        gs_fov_diam_asec (float): Guide Star field of view [arcsec].
        shifts (array-like): WFS shifts [2, no_gs] in meters.
        rotations (array-like): WFS rotations [no_gs] in radians.
        no_subaps (int, optional): Number of sub-apertures on pupil diameter.
            If None, only GS centers are displayed.
        tech_fov_diam_asec (float, optional): Technical FoV diameter [arcsec] used
            to compute meta pupil diameter. If None, ``gs_fov_diam_asec`` is used.
        ngs_fov_diam_asec (float, optional): NGS FoV diameter [arcsec].
        sci_fov_diam_asec (float, optional): Science FoV diameter [arcsec].
        sci_square (bool, optional): If True, draw science FoV as 4 circles on a square.
        title (str, optional): Plot title.
        figsize (tuple, optional): Matplotlib figure size in inches, e.g. ``(8, 8)``.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. If provided, no new figure is created.
        display_sa_lines (bool, optional): If True, draw SA reference grid lines.
        gs_uniform_color (bool, optional): If True, all GS use the same color.
        gs_color (str, optional): Color used when ``gs_uniform_color`` is True.
        gs_alpha (float, optional): Alpha value for GS drawings (useful for overlap view).
        gs_circles_filled (bool, optional): If True, fill GS circles when no_subaps is None.
        show_plot (bool, optional): If True, display figure.
        verbose (bool, optional): If True, print derived geometric values.

    Returns:
        dict: Geometry and plotting data useful for validation/testing.
    """
    if shifts is None:
        shifts = np.zeros((2, int(no_gs)), dtype=float)
    else:
        shifts = np.asarray(shifts, dtype=float)
        if shifts.shape != (2, int(no_gs)):
            raise ValueError('shifts must have shape (2, no_gs).')
    if rotations is None:
        rotations = np.zeros(int(no_gs), dtype=float)
    else:
        rotations = np.asarray(rotations, dtype=float)
        if rotations.shape[0] != int(no_gs):
            raise ValueError('rotations must have length no_gs.')

    no_gs = int(no_gs)

    asec2rad = (np.pi / 180) / 3600.0
    if tech_fov_diam_asec is None:
        tech_fov_diam_asec = gs_fov_diam_asec

    beta_angle = gs_fov_diam_asec / 2.0
    delta_diam = dm_height * np.tan(tech_fov_diam_asec * asec2rad)
    meta_diam = diam + delta_diam
    if np.isfinite(gs_height):
        gs_patch_diam = diam * (gs_height - dm_height) / gs_height
    else:
        gs_patch_diam = diam
    gs_patch_shift = dm_height * np.tan(gs_fov_diam_asec / 2.0 * asec2rad)
    gs_meta_diam = gs_patch_diam + 2.0 * gs_patch_shift
    gs_fov_dm = 2.0 * np.arctan((gs_meta_diam / 2.0 - diam / 2.0) / dm_height) / asec2rad

    ngs_diam = ngs_patch_shift = ngs_meta_diam = None
    if ngs_fov_diam_asec is not None:
        ngs_diam = diam + dm_height * np.tan(ngs_fov_diam_asec * asec2rad)
        ngs_patch_shift = dm_height * np.tan(ngs_fov_diam_asec / 2.0 * asec2rad)
        ngs_meta_diam = ngs_diam + 2.0 * ngs_patch_shift

    sci_diam = sci_patch_shift = sci_meta_diam = None
    if sci_fov_diam_asec is not None:
        sci_diam = diam + dm_height * np.tan(sci_fov_diam_asec * asec2rad)
        sci_patch_shift = dm_height * np.tan(sci_fov_diam_asec / 2.0 * asec2rad)
        sci_meta_diam = sci_diam + 2.0 * sci_patch_shift

    subaps_size = None
    sa_shift = None
    if no_subaps is not None:
        no_subaps = int(no_subaps)
        if no_subaps <= 0:
            raise ValueError('no_subaps must be > 0 when provided.')
        subaps_size = gs_patch_diam / no_subaps
        sa_shift = gs_patch_shift / subaps_size

    if verbose: # pragma: no cover
        print(f'LGS angle:                   {beta_angle:.4f}asec')
        print(f'meta diameter:               {meta_diam:.4f}m')
        print(f'gs patch diameter:           {gs_patch_diam:.4f}m')
        if ngs_fov_diam_asec is not None:
            print(f'ngs patch diameter:          {ngs_diam:.4f}m')
        if sci_fov_diam_asec is not None:
            print(f'sci patch diameter:          {sci_diam:.4f}m')
        print(f'gs patch shift:              {gs_patch_shift:.4f}m')
        if ngs_fov_diam_asec is not None:
            print(f'ngs patch shift:             {ngs_patch_shift:.4f}m')
        if sci_fov_diam_asec is not None:
            print(f'sci patch shift:             {sci_patch_shift:.4f}m')
        print(f'gs meta diameter:            {gs_meta_diam:.4f}m')
        print(f'gs FoV @ DM height:          {gs_fov_dm:.4f}asec')
        if no_subaps is not None:
            print(f'sub-aperture size:           {subaps_size:.4f}m')
            print(f'sub-aperture relative shift: {sa_shift * 100.0:.4f}%')

    # Plot directly in physical units (meters), avoiding IDL-style device scaling.
    shifts_plot = shifts

    if no_subaps is not None:
        sa_size_plot = gs_patch_diam / no_subaps
        circle_size = sa_size_plot / 4.0
        dsa = gs_patch_diam / no_subaps
        axis_1d = (np.arange(no_subaps) - (no_subaps - 1) / 2.0) * dsa
        x, y = np.meshgrid(axis_1d, axis_1d, indexing='ij')
    else:
        sa_size_plot = None
        circle_size = None
        x = y = None

    cmap = plt.get_cmap('tab10')
    if gs_uniform_color:
        colors = [gs_color] * no_gs
    else:
        colors = [cmap(k % 10) for k in range(no_gs)]

    if ax is None:
        if figsize is None:
            figsize = (12.0, 12.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_aspect('equal', adjustable='box')
    max_shift = float(np.max(np.linalg.norm(shifts_plot.T, axis=1))) if no_gs > 0 else 0.0
    radial_extent = gs_patch_shift + max_shift + gs_patch_diam / 2.0
    axis_half_span = max(meta_diam / 2.0, radial_extent) * 1.1
    ax.set_xlim(-axis_half_span, axis_half_span)
    ax.set_ylim(-axis_half_span, axis_half_span)
    ax.set_title('' if title is None else title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    ax.add_patch(
        patches.Circle((0.0, 0.0), radius=meta_diam / 2.0,
                       fill=False, linestyle='--', linewidth=1.0,
                       edgecolor='black', zorder=0)
    )

    if ngs_meta_diam is not None:
        ax.add_patch(
            patches.Circle((0.0, 0.0), radius=ngs_meta_diam / 2.0,
                           fill=False, linewidth=1.2,
                           edgecolor='tab:green', zorder=1)
        )
    if sci_meta_diam is not None:
        if sci_square:
            for i in range(4):
                ang = 2.0 * np.pi / 4.0 * i + np.pi / 4.0
                xsh = sci_patch_shift * np.cos(ang)
                ysh = sci_patch_shift * np.sin(ang)
                ax.add_patch(
                    patches.Circle((xsh, ysh), radius=sci_diam / 2.0,
                                   fill=False, linewidth=1.2,
                                   edgecolor='tab:red', zorder=6)
                )
        else:
            ax.add_patch(
                patches.Circle((0.0, 0.0), radius=sci_meta_diam / 2.0,
                               fill=False, linewidth=1.2,
                               edgecolor='tab:red', zorder=6)
            )

    centers = None
    if no_subaps is not None:
        centers = np.zeros((no_gs, no_subaps, no_subaps, 2), dtype=float)
    gs_centers = np.zeros((no_gs, 2), dtype=float)

    for k in range(no_gs):
        angle = 2.0 * np.pi / no_gs * k
        xsh = gs_patch_shift * np.cos(angle)
        ysh = gs_patch_shift * np.sin(angle)

        if display_sa_lines and no_subaps is not None:
            for i in range(no_subaps):
                xv = x[i, no_subaps // 2] + xsh + shifts_plot[0, k]
                yh = y[no_subaps // 2, i] + ysh + shifts_plot[1, k]
                ax.plot([xv, xv], [-gs_patch_diam / 2.0 + ysh,
                                   gs_patch_diam / 2.0 + ysh],
                    color=colors[k], linewidth=0.5, alpha=gs_alpha)
                ax.plot([-gs_patch_diam / 2.0 + xsh,
                         gs_patch_diam / 2.0 + xsh],
                    [yh, yh], color=colors[k], linewidth=0.5, alpha=gs_alpha)

        c, s = np.cos(rotations[k]), np.sin(rotations[k])
        gs_centers[k, 0] = xsh + shifts_plot[0, k]
        gs_centers[k, 1] = ysh + shifts_plot[1, k]

        if no_subaps is not None:
            xr = x * c - y * s + gs_centers[k, 0]
            yr = y * c + x * s + gs_centers[k, 1]
            centers[k, :, :, 0] = xr
            centers[k, :, :, 1] = yr
            ax.scatter(xr.ravel(), yr.ravel(), s=max(2.0, circle_size * 6.0),
                       color=colors[k], marker='o', alpha=gs_alpha, zorder=2)
        else:
            ax.add_patch(
                patches.Circle(
                    (gs_centers[k, 0], gs_centers[k, 1]),
                    radius=gs_patch_diam / 2.0,
                    fill=bool(gs_circles_filled),
                    facecolor=colors[k] if gs_circles_filled else 'none',
                    edgecolor=colors[k],
                    linewidth=1.2,
                    alpha=gs_alpha,
                    zorder=2,
                )
            )

    ax.text(0.5, 0.97, f'h={round(dm_height)}m', transform=ax.transAxes,
            ha='center', va='top', color='black')

    if show_plot:
        plt.show()

    return {
        'meta_diam': meta_diam,
        'tech_fov_diam_asec': tech_fov_diam_asec,
        'gs_angle_asec': beta_angle,
        'gs_patch_diam': gs_patch_diam,
        'gs_patch_shift': gs_patch_shift,
        'gs_meta_diam': gs_meta_diam,
        'gs_fov_dm': gs_fov_dm,
        'ngs_fov_diam_asec': ngs_fov_diam_asec,
        'ngs_diam': ngs_diam,
        'ngs_patch_shift': ngs_patch_shift,
        'ngs_meta_diam': ngs_meta_diam,
        'sci_fov_diam_asec': sci_fov_diam_asec,
        'sci_square': bool(sci_square),
        'sci_diam': sci_diam,
        'sci_patch_shift': sci_patch_shift,
        'sci_meta_diam': sci_meta_diam,
        'subaps_size': subaps_size,
        'sa_shift': sa_shift,
        'gs_centers': gs_centers,
        'centers': centers,
        'figure': fig,
        'axes': ax,
    }
