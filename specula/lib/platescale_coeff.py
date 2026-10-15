import numpy as np
from specula.log import get_specula_logger

from scipy.linalg import pinv
from specula import cpuArray

def platescale_coeff(dm_list, pixel_pupil):
    """
    Calculate the coefficients required to properly scale the modal amplitude on different
    deformable mirrors to get an accurate platescale correction in an MCAO system

    The plate scale modes (focus and astigmatism) are modes 2-4 of the full DM basis
    (m2c columns if the DM has an m2c, otherwise influence function rows), independently
    of the DM mode selection (start_mode, idx_modes): the basis must be Zernike-first.

    Parameters
    ----------
    dm_list : list
        List of deformable mirror objects
    pixel_pupil : int
        Size of the pupil in pixels
    
    Returns
    -------
    plateScale : dict
        Dictionary containing plate scale parameters
    """
    logger = get_specula_logger(__name__)

    # plate scale modes are 3: focus and 2 astigmatism modes, after tip and tilt
    n_modes_ps = 3
    idx0 = 2

    # dm.modal_ifunc() cannot be used: it only has the modes of the DM input command,
    # and with start_mode > 0 the plate scale modes may not be among them
    def ps_modes(dm):
        # Plate scale modes of the full DM basis
        n_basis = dm.m2c.shape[1] if dm.m2c is not None else dm.ifunc.shape[0]
        if n_basis < idx0 + n_modes_ps:
            raise ValueError(f'The DM basis has {n_basis} modes, at least '
                             f'{idx0 + n_modes_ps} are needed for the plate scale modes')
        missing = [m for m in range(idx0, idx0+n_modes_ps) if m not in dm.selected_modes]
        if missing:
            raise ValueError(f'The plate scale modes {missing} are not driven by the DM input command')
        if dm.m2c is not None:
            return np.dot(cpuArray(dm.m2c[:, idx0:idx0+n_modes_ps]).T, cpuArray(dm.ifunc))
        return cpuArray(dm.ifunc[idx0:idx0+n_modes_ps, :])

    aIfunc = ps_modes(dm_list[0])

    maska = cpuArray(dm_list[0].mask)
    idxa = np.where(maska)
    smaska = maska.shape

    half_p = pixel_pupil // 2

    # Extract pupil region
    pup_mask = maska[smaska[0]//2-half_p:smaska[0]//2+half_p, smaska[1]//2-half_p:smaska[1]//2+half_p]
    mask_indices = np.where(pup_mask)
    n_pup_mask = np.count_nonzero(pup_mask)

    coeff = np.zeros((len(dm_list)-1, n_modes_ps))

    # Process other DMs
    for i in range(1, len(dm_list)):
        bIfunc = ps_modes(dm_list[i])

        maskb = cpuArray(dm_list[i].mask)
        idxb = np.where(maskb)
        smaskb = maskb.shape

        cubea = np.zeros((pixel_pupil, pixel_pupil, n_modes_ps))
        cubeb = np.zeros((pixel_pupil, pixel_pupil, n_modes_ps))
        cubea2D = np.zeros((n_pup_mask, n_modes_ps))
        cubeb2D = np.zeros((n_pup_mask, n_modes_ps))

        for icubes in range(n_modes_ps):
            # Process first DM influence function
            tempa = np.zeros(smaska)
            tempa[idxa[0],idxa[1]] = aIfunc[icubes, :]
            tempa = tempa[smaska[0]//2-half_p:smaska[0]//2+half_p, smaska[1]//2-half_p:smaska[1]//2+half_p]
            tempa = tempa * pup_mask.astype(float)

            tempa[mask_indices] -= np.mean(tempa[mask_indices[0], mask_indices[1]])

            cubea[:, :, icubes] = tempa
            
            cubea2D[:, icubes] = tempa[mask_indices[0], mask_indices[1]]

            # Process second DM influence function
            tempb = np.zeros(smaskb)
            tempb[idxb[0],idxb[1]] = bIfunc[icubes, :]
            tempb = tempb[smaskb[0]//2-half_p:smaskb[0]//2+half_p, smaskb[1]//2-half_p:smaskb[1]//2+half_p]
            tempb = tempb * pup_mask.astype(float)

            # Same operation as for tempa
            tempb[mask_indices[0], mask_indices[1]] -= np.mean(tempb[mask_indices[0], mask_indices[1]])

            cubeb[:, :, icubes] = tempb
            cubeb2D[:, icubes] = tempb[mask_indices[0], mask_indices[1]]

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cubea[:,:,0])
            plt.figure()
            plt.imshow(cubeb[:,:,0])
            plt.show()

        # Calculate projection matrices
        cubea2D_inv = pinv(cubea2D)
        proj = np.dot(cubea2D_inv, cubeb2D)

        # Extract diagonal elements
        proj_diag = np.diag(proj)

        # stack the coefficients
        coeff[i-1, :] = proj_diag

    logger.info(f"plate scale modes amplitude: {np.abs(coeff)}")

    return coeff
