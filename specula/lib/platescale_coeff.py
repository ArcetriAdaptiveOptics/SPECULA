import numpy as np
from scipy.linalg import pinv
from specula import cpuArray

def platescale_coeff(dm_list, start_modes, pixel_pupil, verbose=False):
    """
    Calculate the coefficients required to properly scale the modal amplitude on different
    deformable mirrors to get an accurate platescale correction in an MCAO system
    
    Parameters
    ----------
    dm_list : list
        List of deformable mirror objects
    start_modes : list
        List of starting mode indices for each DM
    pixel_pupil : int
        Size of the pupil in pixels
    verbose : bool
        If True, print additional information
    
    Returns
    -------
    plateScale : dict
        Dictionary containing plate scale parameters
    """

    # plate scale modes are 3: focus and 2 astigmatism modes
    n_modes_ps = 3

    # Extract influence functions from the first DM
    aIfunc = cpuArray(dm_list[0].ifunc)
    # The first DM is the ground DM, so we always skip the tip-tilt modes
    if dm_list[0].m2c is not None:
        aIfunc = np.dot(cpuArray(dm_list[0].m2c[:,2:2+n_modes_ps]).T, aIfunc)
    else:
        aIfunc = aIfunc[2:2+n_modes_ps, :]  # Skip tip-tilt modes

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
        # Skip DMs where focus and astigmatism modes are not present
        if start_modes[i] <= 2:
            # index of focus
            if start_modes[i] == 0:
                idx0 = 2
            else:
                idx0 = 0

            bIfunc = cpuArray(dm_list[i].ifunc)
            if hasattr(dm_list[i], 'm2c') and dm_list[i].m2c is not None:
                bIfunc = np.dot(cpuArray(dm_list[i].m2c[:, idx0:n_modes_ps+idx0]).T, bIfunc)
            else:
                bIfunc = bIfunc[idx0:n_modes_ps+idx0, :]

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
        else:
            proj_diag = np.zeros(n_modes_ps)

        # stack the coefficients
        coeff[i-1, :] = proj_diag

    if verbose:
        print(f"plate scale modes amplitude: {np.abs(coeff)}")

    return coeff
