import numpy as np
from astropy.io import fits
import os

from .utils import radial_order

# def save_correction_vector(dir_path:str,min_corr:float,max_corr:float,Nmodes:int=660,Ncorrmodes:int=None):
#     if Ncorrmodes is None:
#         Ncorrmodes = Nmodes
#     max_rad_order = radial_order(Nmodes)+1
#     cc = np.linspace(max_corr,min_corr,max_rad_order-2)
#     tt = np.hstack([np.repeat(cc[i-2],i) for i in range(2,max_rad_order)])
#     residuals = np.zeros(Nmodes)
#     residuals[:Ncorrmodes] = tt[:Ncorrmodes]
    
#     dirpath = os.path.join(dir_path,'data')
#     os.makedirs(dirpath,exist_ok=True)
#     fname = f'correction_vector_{Ncorrmodes}modes_c{max_corr:1.2f}-{min_corr:1.2f}.fits'
#     filepath = os.path.join(dirpath,fname)
#     hdr = fits.Header()
#     hdr['VERSION'] = 1
#     hdr['OBJ_TYPE'] = 'BaseValue'
#     hdr['NDARRAY'] = 1
#     fits.writeto(filepath, residuals, hdr, overwrite=True)
#     print(f'Saved correction vector as {fname}')
#     return fname


def save_correction_vector(dir_path:str, max_corr: float, min_corr: float,  Nmodes: int = 660, Ncorrmodes: int = None):
    """
    Generates a correction vector with logarithmic scaling to maintain 
    constant power-law slopes in residual turbulence PSDs.
    """
    if Ncorrmodes is None:
        Ncorrmodes = Nmodes
    max_rad_order = radial_order(Nmodes)+1
    max_leak = 1.0 - min_corr
    min_leak = 1.0 - max_corr
    leakage_per_order = np.linspace(min_leak, max_leak, max_rad_order - 2)
    cc = 1.0 - leakage_per_order
    tt = np.hstack([np.repeat(cc[i-2], i) for i in range(2, max_rad_order)])
    residuals = np.zeros(Nmodes)
    length_to_fill = min(len(tt), Ncorrmodes)
    residuals[:length_to_fill] = tt[:length_to_fill]
    if Ncorrmodes > length_to_fill:
        residuals[length_to_fill:Ncorrmodes] = min_corr

    os.makedirs(dir_path, exist_ok=True)
    fname = f'correction_vector_{Ncorrmodes}modes_c{max_corr:1.2f}-{min_corr:1.2f}.fits'
    filepath = os.path.join(dir_path,'data', fname)
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['OBJ_TYPE'] = 'BaseValue'
    hdr['MAX_CORR'] = max_corr
    hdr['MIN_CORR'] = min_corr
    
    fits.writeto(filepath, residuals, hdr, overwrite=True)
    print(f'✅ Saved: {fname}')
    return fname
    
if __name__ == "__main__":
    dir_path = '/raid1/mmenessini/calibration/SOUL'
    Ncorrmodes = 500
    save_correction_vector(dir_path=dir_path, max_corr=0.99, min_corr=0.2, Ncorrmodes=Ncorrmodes)
    save_correction_vector(dir_path=dir_path, max_corr=0.9, min_corr=0.2, Ncorrmodes=Ncorrmodes)
    save_correction_vector(dir_path=dir_path, max_corr=0.85, min_corr=0.2, Ncorrmodes=Ncorrmodes)
    save_correction_vector(dir_path=dir_path, max_corr=0.8, min_corr=0.2, Ncorrmodes=Ncorrmodes)