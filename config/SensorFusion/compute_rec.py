from astropy.io import fits
from specula import cpuArray
import numpy as np

def compute_rec(im_tag:str, Nmodes:int):
    hdul = fits.open('./calibration/im/'+im_tag+'_im.fits')
    intmat = hdul[1].data.copy()
    U,S,Vt = np.linalg.svd(intmat[:,:Nmodes],full_matrices=False)
    rec = (Vt.T * 1/S) @ U.T
    return rec

def save_rec(rec, rec_tag:str, overwrite:bool=True):
    filename = './calibration/rec/'+rec_tag+'_rec.fits'
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

if __name__ == "__main__":

    Nmodes = 1200
    im_tag = 'dm_1851modes'
    rec_tag = f'dm_{Nmodes:1.0f}modes'

    rec = compute_rec(im_tag=im_tag, Nmodes=Nmodes)
    save_rec(rec, rec_tag=rec_tag)