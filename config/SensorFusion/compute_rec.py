from astropy.io import fits
from specula import cpuArray
import numpy as np
import os

def compute_rec(im_tag:str, Nmodes:int):
    hdul = fits.open('./calibration/im/'+im_tag+'_im.fits')
    intmat = hdul[1].data.copy()
    D = intmat[:,:Nmodes]
    U,S,Vt = np.linalg.svd(D,full_matrices=False)
    rec = (Vt.T * 1/S) @ U.T
    return rec

def compute_ml_rec(im_tag:str, Nmodes:int, sn_tag:str=None, RON:float=None):    
    im_hdul = fits.open('./calibration/im/'+im_tag+'_im.fits')
    intmat = im_hdul[1].data.copy()
    D = intmat[:,:Nmodes]
    sn_hdul = fits.open('./calibration/slopenull/'+sn_tag+'_im.fits')
    slope_null = sn_hdul[1].data.copy()
    noise_cov = slope_null + RON
    flux = np.sum(slope_null)
    noise_cov /= flux
    DtCn = D.T @ np.diag(1/noise_cov)
    rec = np.linalg.pinv(DtCn @ D) @ DtCn
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


def compute_pyr_rec(Nmodes:int, compute_ml:bool=False, sn_tag = 'pyr_slope_null'):
    im_tag = 'pyr_1851modes'
    if compute_ml is False:
        rec_tag = f'pyr_{Nmodes:1.0f}modes'
        rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    else:
        rec_tag = f'pyr_{Nmodes:1.0f}modes_ml'
        rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, sn_tag=sn_tag, RON=0.5)
    return rec, rec_tag


def compute_zwfs_rec(Nmodes:int, compute_ml:bool=False, sn_tag = 'zwfs_slope_null'):
    im_tag = 'zwfs_1851modes'
    if compute_ml is False:
        rec_tag = f'zwfs_{Nmodes:1.0f}modes'
        rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    else:
        rec_tag = f'zwfs_{Nmodes:1.0f}modes_ml'
        rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, sn_tag=sn_tag, RON=0.5)
    return rec, rec_tag


if __name__ == "__main__":

    rec,rec_tag = compute_pyr_rec(Nmodes=1200)
    save_rec(rec, rec_tag=rec_tag)

    rec,rec_tag = compute_zwfs_rec(Nmodes=100)
    save_rec(rec, rec_tag=rec_tag)