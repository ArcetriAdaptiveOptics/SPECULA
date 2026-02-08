"""
Utility functions for SynIM-based interaction matrix computation.
"""

import synim.synim as synim
from specula import cpuArray, np


def compute_im_synim(misreg_params,
                     pup_diam_m,
                     pup_mask,
                     ifunc_3d,
                     dm_mask,
                     source_polar_coords,
                     source_height,
                     wfs_nsubaps,
                     wfs_fov_arcsec,
                     idx_valid_sa,
                     apply_absolute_slopes=False,
                     verbose=False):
    """
    Compute interaction matrix using SynIM with mis-registration parameters.
    
    Parameters
    ----------
    misreg_params : array_like, shape (4,) or (6,)
        Mis-registration parameters:
        [shift_x, shift_y, rotation, magnification(, magn_x, magn_y)]
    pup_diam_m : float
        Pupil diameter [m]
    pup_mask : ndarray
        Pupil mask
    ifunc_3d : ndarray
        3D influence functions array
    dm_mask : ndarray
        DM mask
    source_polar_coords : tuple
        Source polar coordinates (theta, phi)
    source_height : float
        Source height [m] (inf for NGS)
    wfs_nsubaps : int
        Number of subapertures across diameter
    wfs_fov_arcsec : float
        Subaperture field of view [arcsec]
    idx_valid_sa : ndarray
        Valid subaperture indices
    apply_absolute_slopes : bool
        Apply absolute value to slopes
    verbose : bool
        Print debug info
    
    Returns
    -------
    im : ndarray, shape (nslopes, nmodes)
        Interaction matrix
    """

    # Extract parameters
    shift_x = float(misreg_params[0])
    shift_y = float(misreg_params[1])
    rotation = float(misreg_params[2])
    magnification = 1.0 + float(misreg_params[3])
    if len(misreg_params) == 6:
        wfs_anamorphosis_90 = float(misreg_params[4])
        wfs_anamorphosis_45 = float(misreg_params[5])
    else:
        wfs_anamorphosis_90 = 1.0
        wfs_anamorphosis_45 = 1.0

    # Get source parameters
    gs_pol_coo = tuple(cpuArray(source_polar_coords))
    gs_height = source_height if source_height != float('inf') else float('inf')

    if verbose:
        print(f"  Computing IM with SynIM:")
        print(f"    shift_x={shift_x:.3f} px, shift_y={shift_y:.3f} px")
        print(f"    rotation={rotation:.3f} deg, magnification={magnification:.6f}")
        if len(misreg_params) == 6:
            print(f"    anamorphosis_90={wfs_anamorphosis_90:.6f},"
                  f"anamorphosis_45={wfs_anamorphosis_45:.6f}")

    # Compute IM with SynIM
    im = synim.interaction_matrix(
        pup_diam_m=pup_diam_m,
        pup_mask=pup_mask,
        dm_array=cpuArray(ifunc_3d),
        dm_mask=cpuArray(dm_mask).T,
        dm_height=0.0,
        dm_rotation=0.0,
        gs_pol_coo=gs_pol_coo,
        gs_height=gs_height,
        wfs_nsubaps=wfs_nsubaps,
        wfs_rotation=rotation,
        wfs_translation=(shift_x, shift_y),
        wfs_mag_global=magnification,
        wfs_anamorphosis_90=wfs_anamorphosis_90,
        wfs_anamorphosis_45=wfs_anamorphosis_45,
        wfs_fov_arcsec=wfs_fov_arcsec,
        idx_valid_sa=idx_valid_sa,
        verbose=False,
        specula_convention=True
    )

    if apply_absolute_slopes:
        im = np.abs(im)

    return im
