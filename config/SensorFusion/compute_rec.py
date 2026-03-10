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
    sn_hdul = fits.open('./calibration/slopenulls/'+sn_tag+'.fits')
    slope_null = sn_hdul[1].data.copy()
    flux = np.sum(slope_null)
    noise_cov = slope_null + RON/flux
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


def compute_pyr_rec(Nmodes:int, im_tag:str='pyr_1851modes', compute_ml:bool=False, sn_tag = 'pyr_slope_null'):
    if compute_ml is False:
        rec_tag = f'pyr_{Nmodes:1.0f}modes'
        rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    else:
        rec_tag = f'pyr_{Nmodes:1.0f}modes_ml'
        rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, sn_tag=sn_tag, RON=0.5)
    return rec, rec_tag


def compute_zwfs_rec(Nmodes:int, im_tag:str='zwfs_1851modes', compute_ml:bool=False, sn_tag = 'zwfs_slope_null'):
    if compute_ml is False:
        rec_tag = f'zwfs_{Nmodes:1.0f}modes'
        rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    else:
        rec_tag = f'zwfs_{Nmodes:1.0f}modes_ml'
        rec = compute_ml_rec(im_tag=im_tag, Nmodes=Nmodes, sn_tag=sn_tag, RON=0.5)
    return rec, rec_tag


if __name__ == "__main__":

    Nmodes = 1200
    # rMods = np.array([0,0.5,1,3,4,6])
    # for rMod in rMods:
    #     rec,_ = compute_pyr_rec(Nmodes=Nmodes,im_tag=f'pyr{rMod:1.1f}_1851modes')
    #     save_rec(rec, rec_tag=f'pyr{rMod:1.1f}_{Nmodes:1.0f}modes')

    dotSizes = np.array([1,1.5,2])
    for dotSize in dotSizes:
        rec,_ = compute_zwfs_rec(Nmodes=Nmodes,im_tag=f'z{dotSize:1.1f}wfs_1851modes')
        save_rec(rec, rec_tag=f'z{dotSize:1.1f}wfs_{Nmodes:1.0f}modes')

        rec,_ = compute_zwfs_rec(Nmodes=Nmodes,compute_ml=True,im_tag=f'z{dotSize:1.1f}wfs_1851modes',sn_tag='z1.0wfs_slope_null')
        save_rec(rec, rec_tag=f'z{dotSize:1.1f}wfs_{Nmodes:1.0f}modes_ml')