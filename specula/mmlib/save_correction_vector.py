import numpy as np
from astropy.io import fits
import os

def save_correction_vector(min_corr:float,max_corr:float,
                        max_rad_order:int=36,Nmodes:int=660,
                        Ncorrmodes:int=None):
    if Ncorrmodes is None:
        Ncorrmodes = Nmodes
    cc = np.linspace(max_corr,min_corr,max_rad_order-2)
    tt = np.hstack([np.repeat(cc[i-2],i) for i in range(2,max_rad_order)])
    residuals = np.zeros(Nmodes)
    residuals[:Ncorrmodes] = tt[:Ncorrmodes]
    
    dirpath = '/raid1/mmenessini/calibration/SOUL/data/'
    os.makedirs(dirpath,exist_ok=True)
    fname = f'correction_vector_{Ncorrmodes}modes_c{max_corr:1.2f}.fits'
    filepath = os.path.join(dirpath,fname)
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['OBJ_TYPE'] = 'BaseValue'
    hdr['NDARRAY'] = 1
    fits.writeto(filepath, residuals, hdr, overwrite=False)
    print(f'Saved correction vector as {fname}')


if __name__ == "__main__":
    Ncorrmodes = 500
    save_correction_vector(max_corr=0.99,min_corr=0.2,Ncorrmodes=Ncorrmodes)
    save_correction_vector(max_corr=0.9,min_corr=0.2,Ncorrmodes=Ncorrmodes)
    save_correction_vector(max_corr=0.85,min_corr=0.2,Ncorrmodes=Ncorrmodes)
    save_correction_vector(max_corr=0.8,min_corr=0.2,Ncorrmodes=Ncorrmodes)