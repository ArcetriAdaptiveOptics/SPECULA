from astropy.io import fits
from specula import cpuArray
import numpy as np
import os

from specula.mmlib.utils import von_karman_power, radial_order
from specula.lib.mmse_reconstructor import compute_mmse_reconstructor


def compute_and_save_rec(root_dir:str, im_tag:str, rec_tag:str, Nmodes:int, 
                ml:bool=False, slope_null=None, RON:float=0.0, r0=None, L0=25,
                mmse:bool=False, diam:float=None, overwrite:bool=False):
    print(rec_tag,im_tag)
    rec = compute_rec(root_dir, im_tag, Nmodes, ml=ml, slope_null=slope_null, RON=RON, mmse=mmse, diam=diam, r0=r0, L0=L0)
    save_rec(root_dir, rec, rec_tag, overwrite=overwrite)


def compute_rec(root_dir:str, im_tag:str, Nmodes:int, 
                ml:bool=False, slope_null=None, RON:float=0.0, 
                mmse:bool=False, diam:float=None, r0=None, L0=None):    
    im_hdul = fits.open(os.path.join(root_dir,'im',im_tag+'.fits'))
    intmat = im_hdul[1].data.copy()
    D = intmat[:,:Nmodes]
    if ml or mmse:
        noise_cov = np.diag((slope_null + RON))
        if mmse:
            k = radial_order(np.arange(Nmodes))/diam
            turb_cov = np.diag(np.sqrt(von_karman_power(k,r0=r0,L0=L0,D=diam))*(2*np.pi*500))**2
        else:
            turb_cov = np.zeros([Nmodes, Nmodes])
        rec = compute_mmse_reconstructor(interaction_matrix=D, c_atm=turb_cov, c_noise=noise_cov, verbose=True, xp=np, dtype=np.float64)
        # DtCn = D.T @ np.diag(1/(slope_null + RON))
        # rec = np.linalg.pinv(DtCn @ D) @ DtCn
    else:
        U,S,Vt = np.linalg.svd(D,full_matrices=False)
        rec = (Vt.T * 1/S) @ U.T
    return rec


def save_rec(root_dir:str, rec, rec_tag:str, overwrite:bool=False):
    path = os.path.join(root_dir,'rec')
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path,rec_tag+'_rec.fits')
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['PUP_TAG'] = ''
    hdr['SA_TAG'] = ''
    hdr['NORMFACT'] = 0.0
    hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
    hdul = fits.HDUList([hdu])
    hdul.append(fits.ImageHDU(data=cpuArray(rec), name='REC'))
    hdul.writeto(filename, overwrite=overwrite)
    hdul.close()
    print('Reconstructor saved as '+rec_tag)


# def compute_pyr_rec(Nmodes:int, im_tag:str='pyr_1821modes', compute_ml:bool=False, frame_tag = '', cov_tag=None):
#     if compute_ml is False:
#         rec_tag = f'pyr_{Nmodes:1.0f}modes'
#         rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
#     else:
#         rec_tag = f'pyr_{Nmodes:1.0f}modes_ml'
#         rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, frame_tag=frame_tag, cov_tag=cov_tag, RON=0.5, isPyr=True)
#     return rec, rec_tag


# def compute_zwfs_rec(Nmodes:int, im_tag:str='zwfs_1821modes', compute_ml:bool=False, frame_tag = '', cov_tag=None):
#     if compute_ml is False:
#         rec_tag = f'zwfs_{Nmodes:1.0f}modes'
#         rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
#     else:
#         rec_tag = f'zwfs_{Nmodes:1.0f}modes_ml'
#         rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, frame_tag=frame_tag, cov_tag=cov_tag, RON=0.5, isPyr=False)
#     return rec, rec_tag


if __name__ == "__main__":

    Nmodes = 440
    root_dir = '/raid1/mmenessini/calibration/EKARUS'
    im_tag = 'pyr5.0_40x40_im'
    rec_tag = 'pyr5.0_40x40_440modes'
    compute_and_save_rec(root_dir=root_dir, im_tag=im_tag, rec_tag=rec_tag, Nmodes=Nmodes, overwrite=True)


#     Nmodes = 1300
#     rMods = np.array([0,0.5,1,2,3])
#     for rMod in rMods:
#         rec,_ = compute_pyr_rec(Nmodes=Nmodes,im_tag=f'pyr{rMod:1.1f}_1821modes')
#         save_rec(rec, rec_tag=f'pyr{rMod:1.1f}_{Nmodes:1.0f}modes')

#     # rec,_ = compute_pyr_rec(Nmodes=Nmodes,compute_ml=True, cov_tag=None, 
#     #                         im_tag=f'pyr0.0_1821modes',frame_tag=f'pyr0.0_frame')
#     # save_rec(rec, rec_tag=f'pyr0.0_{Nmodes:1.0f}modes_ml')

#     rec,_ = compute_pyr_rec(Nmodes=150,im_tag=f'pyr0.0_16x16_239modes')
#     save_rec(rec, rec_tag=f'pyr0.0_16x16_150modes')

#     dotSizes = np.array([1,1.5,2])
#     for dotSize in dotSizes:
#         rec,_ = compute_zwfs_rec(Nmodes=Nmodes,im_tag=f'z{dotSize:1.1f}wfs_1821modes')
#         save_rec(rec, rec_tag=f'z{dotSize:1.1f}wfs_{Nmodes:1.0f}modes')

#         # rec,_ = compute_zwfs_rec(Nmodes=Nmodes,compute_ml=True, cov_tag=None, #'bmc2k_vlt', #cov_tag=None,
#         #                          im_tag=f'z{dotSize:1.1f}wfs_1821modes',frame_tag=f'z{dotSize:1.1f}wfs_frame')
#         # save_rec(rec, rec_tag=f'z{dotSize:1.1f}wfs_{Nmodes:1.0f}modes_ml')
