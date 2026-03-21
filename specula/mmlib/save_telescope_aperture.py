import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)

from astropy.io import fits
from specula.lib.toccd import toccd
import os

import matplotlib.pyplot as plt

from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

def save_pupil_to_size(tag:str, Npix:int, thr:float=0.5, D:float=8.2):
    hdu = fits.open('./telescope/'+tag+'_512pixels.fits')
    data = hdu[0].data
    pupil = data[:,:,0]
    new_pupil = toccd(pupil,(Npix,Npix),xp=specula.xp)
    new_pupil = new_pupil >= thr*new_pupil.max()
    # new_pupil = specula.xp.array(new_pupil,dtype=specula.float_dtype)

    os.makedirs('./calibration/pupilstop/',exist_ok=True)
    fname = './calibration/pupilstop/'+tag+f'_{Npix:1.0f}pixels.fits'
    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=new_pupil)
    pupilstop.save(fname)
    return new_pupil

if __name__ == "__main__":
    tag = 'vlt_pupil'
    Npix = 160
    aperture=save_pupil_to_size(tag, Npix,thr=0.69)
    plt.figure()
    plt.imshow(aperture,origin='lower',cmap='gray')
    plt.show()


