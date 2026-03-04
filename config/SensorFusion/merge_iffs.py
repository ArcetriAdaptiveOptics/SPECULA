from astropy.io import fits
import numpy as np
from specula.lib.make_mask import make_mask
from specula import cpuArray
import matplotlib.pyplot as plt

import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft

def filter_iffs(iffs, coeff2modes):
    IFFs = iffs.copy()
    modes2coeff = np.linalg.pinv(coeff2modes)
    coeffs = modes2coeff @ IFFs
    fit_modes = coeff2modes @ coeffs
    IFFs -= fit_modes
    return IFFs

def combine_influence_functions(N:int, if1_tag:str, if2_tag:str,
                                 pupil_pixels:int=120, oversampling:int=4,
                                r0:float=10e-2, L0:float=25, telescope_diameter:float=8.2,
                                obsratio:float=0.0, diaratio:float=1.0):
    
    # Load files
    kl1_inv_hdu = fits.open('./calibration/ifunc/'+if1_tag+'_kl_inv.fits')
    kl1_basis = np.linalg.pinv(kl1_inv_hdu[1].data)
    kl2_inv_hdu = fits.open('./calibration/ifunc/'+if2_tag+'_kl_inv.fits')
    kl2_basis = np.linalg.pinv(kl2_inv_hdu[1].data)
    if kl1_basis.shape[1] > kl2_basis.shape[1]:
        modes = kl2_basis[:,:N]
    else:
        modes = kl1_basis[:,:N]
    if1_hdu = fits.open('./calibration/ifunc/'+if1_tag+'_ifunc.fits')
    if1 = if1_hdu[1].data
    if2_hdu = fits.open('./calibration/ifunc/'+if2_tag+'_ifunc.fits')
    if2 = if2_hdu[1].data
    
    pupil_mask = make_mask(pupil_pixels, obsratio, diaratio, xp=specula.xp)
    
    if1_filt = filter_iffs(if1,modes)
    if2_filt = filter_iffs(if2,modes)

    influence_functions = np.hstack([if1_filt,if2_filt]).T

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=pupil_mask,
        diameter=telescope_diameter,
        influence_functions=influence_functions,
        r0=r0,
        L0=L0,
        verbose=True,
        zern_modes=0,
        oversampling=oversampling,
        if_max_condition_number=1e+2,
        xp=specula.xp,
        dtype = specula.xp.float32)
    
    kl_basis = np.vstack([modes.T,kl_basis])

    S1 = np.linalg.svd(kl1_basis,compute_uv=False) #np.diag(kl1_basis.T @ kl1_basis)   
    S2 = np.linalg.svd(kl2_basis,compute_uv=False) #np.diag(kl2_basis.T @ kl2_basis)    
    S = np.linalg.svd(kl_basis,compute_uv=False) #np.diag(kl_basis @ kl_basis.T)

    plt.figure(figsize=(10, 6))
    plt.plot(cpuArray(S), 'o-', label=f'Combined KL Covariance')
    plt.plot(cpuArray(S1), 'o-', label=f'{if1_tag} KL Covariance')
    plt.plot(cpuArray(S2), 'o-', label=f'{if2_tag} KL Covariance')
    plt.xlabel('Mode number')
    plt.ylabel('Singular value')
    plt.legend()
    plt.grid(True)

    # move to CPU / numpy for plotting if required
    kl_basis = cpuArray(kl_basis)
    pupil_mask = cpuArray(pupil_mask)

    # Plot some modes
    max_modes = min(20, kl_basis.shape[0])

    # Create a mask array for display
    mode_display = np.zeros((max_modes, pupil_mask.shape[0], pupil_mask.shape[1]))

    # Place each mode vector into the 2D pupil shape
    idx_mask = np.where(pupil_mask)
    mode_ids = np.zeros(max_modes,dtype=int)
    for i in range(max_modes//2):
        mode_img = np.zeros(pupil_mask.shape)
        mode_ids[i] = i+1
        mode_img[idx_mask] = kl_basis[i]
        mode_display[i] = mode_img
    for i in range(max_modes//2,max_modes):
        mode_img = np.zeros(pupil_mask.shape)
        mode_ids[i] = kl_basis.shape[0]-max_modes+i
        mode_img[idx_mask] = kl_basis[mode_ids[i]]
        mode_display[i] = mode_img

    # Plot the reshaped modes
    n_rows = int(np.round(np.sqrt(max_modes)))
    n_cols = int(np.ceil(max_modes / n_rows))
    plt.figure(figsize=(18, 12))
    for i in range(max_modes):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(np.ma.masked_array(mode_display[i],mask=1-pupil_mask),origin='lower',cmap='RdBu')
        plt.title(f'Mode {mode_ids[i]}')
        plt.axis('off')
    plt.tight_layout()

    plt.show()
    

    
if __name__ == "__main__":
    combine_influence_functions(N=50, if1_tag='dsm', if2_tag='bmc1k')