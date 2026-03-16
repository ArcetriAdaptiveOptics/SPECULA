from astropy.io import fits
from specula import cpuArray
import numpy as np
import os
from specula.lib.make_mask import make_mask

from specula.lib.mmse_reconstructor import compute_mmse_reconstructor


def radial_order(i_mode):
    noll = i_mode + 2
    return np.ceil(-3.0/2.0+np.sqrt(1+8*noll)/2.0)

def von_karman_power(k,r0,L0,D):
    C = 0.02289558710855519
    B = k**2 + (D/L0)**2
    return C * (r0/D)**(-5.0/3.0) * B**(-11.0/6.0)

def get_mask(pyr:bool=True):
    npix = 120
    if pyr:
        np_size = (npix,npix)
        pup_hdu = fits.open('./calibration/pupils/pyr_pupdata.fits')
        rad = pup_hdu[2].data
        pup_ids = pup_hdu[1].data
        wfs_mask = np.zeros(np_size)
        for j in range(len(rad)):
            f = np.zeros(npix**2)
            np.put(f, pup_ids[:,j], 1)
            f2d = f.reshape(np_size)
            wfs_mask += f2d
    else:
        wfs_mask = make_mask(np_size=npix, diaratio = 48/npix, obsratio=0.0)
    return wfs_mask.astype(bool)


def compute_rec(im_tag:str, Nmodes:int):
    hdul = fits.open('./calibration/im/'+im_tag+'_im.fits')
    intmat = hdul[1].data.copy()
    D = intmat[:,:Nmodes]
    U,S,Vt = np.linalg.svd(D,full_matrices=False)
    rec = (Vt.T * 1/S) @ U.T
    return rec

def compute_ml_rec(im_tag:str, Nmodes:int, frame_tag:str, cov_tag:str=None, RON:float=0.0, isPyr:bool=True):    
    im_hdul = fits.open('./calibration/im/'+im_tag+'_im.fits')
    intmat = im_hdul[1].data.copy()
    D = intmat[:,:Nmodes]
    frame_hdul = fits.open('./calibration/slopenulls/'+frame_tag+'.fits')
    frame_null = frame_hdul[0].data[0]
    wfs_mask = get_mask(pyr=isPyr)
    slope_null = frame_null[wfs_mask]
    noise_cov = np.diag((slope_null + RON))
    if cov_tag is not None:
        diam=8.2
        k = radial_order(np.arange(Nmodes))/diam
        turb_cov = np.diag(np.sqrt(von_karman_power(k,r0=10e-2,L0=25,D=diam))*(2*np.pi*500))**2
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(k,np.diag(turb_cov))
        # plt.grid()
        # plt.show()
    else:
        turb_cov = np.eye(Nmodes)
    rec = compute_mmse_reconstructor(interaction_matrix=D, c_atm=turb_cov, c_noise=noise_cov, verbose=True, xp=np, dtype=np.float64)
    # DtCn = D.T @ np.diag(1/(slope_null + RON))
    # rec = np.linalg.pinv(DtCn @ D) @ DtCn
    return rec


def save_rec(rec, rec_tag:str, overwrite:bool=True):
    path = './calibration/rec/'
    if not os.path.exists(path):
        os.mkdir(path)
    filename = path+rec_tag+'_rec.fits'
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
    print('Reconstructor saved as '+rec_tag+'_rec')


def compute_pyr_rec(Nmodes:int, im_tag:str='pyr_1821modes', compute_ml:bool=False, frame_tag = '', cov_tag=None):
    if compute_ml is False:
        rec_tag = f'pyr_{Nmodes:1.0f}modes'
        rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    else:
        rec_tag = f'pyr_{Nmodes:1.0f}modes_ml'
        rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, frame_tag=frame_tag, cov_tag=cov_tag, RON=0.5, isPyr=True)
    return rec, rec_tag


def compute_zwfs_rec(Nmodes:int, im_tag:str='zwfs_1821modes', compute_ml:bool=False, frame_tag = '', cov_tag=None):
    if compute_ml is False:
        rec_tag = f'zwfs_{Nmodes:1.0f}modes'
        rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    else:
        rec_tag = f'zwfs_{Nmodes:1.0f}modes_ml'
        rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, frame_tag=frame_tag, cov_tag=cov_tag, RON=0.5, isPyr=False)
    return rec, rec_tag


if __name__ == "__main__":

    Nmodes = 1200
    # rMods = np.array([0,0.5,1,2,3])
    # for rMod in rMods:
    #     rec,_ = compute_pyr_rec(Nmodes=Nmodes,im_tag=f'pyr{rMod:1.1f}_1821modes')
    #     save_rec(rec, rec_tag=f'pyr{rMod:1.1f}_{Nmodes:1.0f}modes')

    rec,_ = compute_pyr_rec(Nmodes=Nmodes,compute_ml=True, cov_tag=None, 
                            im_tag=f'pyr0.0_1821modes',frame_tag=f'pyr0.0_frame')
    save_rec(rec, rec_tag=f'pyr0.0_{Nmodes:1.0f}modes_ml')

    rec,_ = compute_pyr_rec(Nmodes=150,im_tag=f'pyr3.0_16x16_239modes')
    save_rec(rec, rec_tag=f'pyr3.0_16x16_150modes')

    # dotSizes = np.array([1,1.5,2])
    # for dotSize in dotSizes:
    #     rec,_ = compute_zwfs_rec(Nmodes=Nmodes,im_tag=f'z{dotSize:1.1f}wfs_1821modes')
    #     save_rec(rec, rec_tag=f'z{dotSize:1.1f}wfs_{Nmodes:1.0f}modes')

        # rec,_ = compute_zwfs_rec(Nmodes=Nmodes,compute_ml=True, cov_tag=None, #'bmc2k_vlt', #cov_tag=None,
        #                          im_tag=f'z{dotSize:1.1f}wfs_1821modes',frame_tag=f'z{dotSize:1.1f}wfs_frame')
        # save_rec(rec, rec_tag=f'z{dotSize:1.1f}wfs_{Nmodes:1.0f}modes_ml')