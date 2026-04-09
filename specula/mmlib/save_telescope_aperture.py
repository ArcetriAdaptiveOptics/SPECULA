import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)

from astropy.io import fits
from specula.lib.toccd import toccd
import os

import matplotlib.pyplot as plt

from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from specula.lib.make_mask import make_mask

def save_copernico_pupil_to_size(destination_dir:str, tag:str, Npix:int, D:float=1.82):
    new_pupil = make_mask(np_size=Npix,obsratio=0.3,spider=True,n_petals=4,angle_offset=15,spider_width=0.02/D)
    os.makedirs(destination_dir,exist_ok=True)
    fname = os.path.join(destination_dir, tag+f'_{Npix:1.0f}pixels.fits')
    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=new_pupil)
    pupilstop.save(fname)
    return new_pupil

def save_pupil_to_size(data_dir:str, destination_dir:str, tag:str, Npix:int, thr:float=0.5, D:float=8.2):
    hdu = fits.open(os.path.join(data_dir, tag+'_512pixels.fits'))
    data = hdu[0].data
    pupil = data[:,:,0]
    new_pupil = toccd(pupil,(Npix,Npix),xp=specula.xp)
    new_pupil = new_pupil >= thr*new_pupil.max()
    # new_pupil = specula.xp.array(new_pupil,dtype=specula.float_dtype)

    os.makedirs(destination_dir,exist_ok=True)
    fname = os.path.join(destination_dir, tag+f'_{Npix:1.0f}pixels.fits')
    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=new_pupil)
    pupilstop.save(fname)
    return new_pupil

if __name__ == "__main__":

    # data_dir = '/raid1/mmenessini/calibration/VLT'
    # destination_dir = '/raid1/mmenessini/calibration/XAO/pupilstop'
    # tag = 'vlt_pupil'
    # Npix = 160
    # aperture=save_pupil_to_size(data_dir, destination_dir, tag, Npix, thr=0.69)
    # plt.figure()
    # plt.imshow(aperture,origin='lower',cmap='gray')
    # plt.show()

    destination_dir = '/raid1/mmenessini/calibration/EKARUS/pupilstop'
    tag = 'copernico_pupil'
    Npix = 120
    aperture=save_copernico_pupil_to_size(destination_dir, tag, Npix)
    plt.figure()
    plt.imshow(aperture,origin='lower',cmap='gray')
    plt.show()


