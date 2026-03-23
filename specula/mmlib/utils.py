import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import welch

import specula
specula.init(-1)  # Default target device

from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.iir_filter_data import IirFilterData



def radial_order(i_mode):
    noll = i_mode + 2
    return np.ceil(-3.0/2.0+np.sqrt(1+8*noll)/2.0)

def von_karman_power(k,r0,L0,D):
    C = 0.02289558710855519
    B = k**2 + (D/L0)**2
    return C * (r0/D)**(-5.0/3.0) * B**(-11.0/6.0)

def get_pupil_mask(npix:int, filepath:str='', pyr:bool=True, pupdiam=None, obsratio=0.0):
    if pyr:
        np_size = (npix,npix)
        pup_hdu = fits.open(filepath)
        rad = pup_hdu[2].data
        pup_ids = pup_hdu[1].data
        wfs_mask = np.zeros(np_size)
        for j in range(len(rad)):
            f = np.zeros(npix**2)
            np.put(f, pup_ids[:,j], 1)
            f2d = f.reshape(np_size)
            wfs_mask += f2d
    else:
        wfs_mask = make_mask(np_size=npix, diaratio = pupdiam/npix, obsratio=obsratio)
    return wfs_mask.astype(bool)


def get_psd(data, dt:float, nperseg:int=1024):
    f,psd=welch(data,fs=1/dt,nperseg=1024,scaling='density',axis=-1)
    return psd,f


def get_reference_psf(root_dir:str,pupil_tag:str='vlt_pupil_160pixels',nd:int=4):
    hdu = fits.open(os.path.join(root_dir,'pupilstop',pupil_tag+'.fits'))
    pupil = hdu[1].data
    psf = calc_psf(np.zeros_like(pupil),pupil.astype(float), imwidth=int(pupil.shape[1]*nd), normalize=True,
                                            xp=np, complex_dtype=np.complex128)
    return psf

def read_freq(params_path:str, obj_name:str=None):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
        if obj_name is None:
            fs = 1.0/float(params['main']['time_step'])
        else:
            fs = 1.0/float(params[obj_name]['dt'])
    return fs

def get_control_data(root_dir:str,controller_name:str, params):
    delay_frames = 1.0 + float(params[controller_name]['delay'])
    if params[controller_name]['class'] == 'IirFilter':
        gain = float(params[controller_name]['iir_gain'])
        iir_path = os.path.join(root_dir,'filter/',str(params[controller_name]['iir_filter_data_object'])+'.fits')
        filter_data_complex = IirFilterData.restore(iir_path)
        filter_data_complex.num *= gain
    else:
        gain = np.array(params[controller_name]['int_gain'])
        try:
            ff = np.array(params[controller_name]['ff'])
        except KeyError:
            ff = np.array([1.0])
        filter_data_complex = IirFilterData.from_gain_and_ff(gain=gain.tolist(),ff=ff.tolist())
    return filter_data_complex, delay_frames



def show_psf(psf, oversampling:int=4, title:str='', ext=0.25, vmin=-8, vmax=0, cmap='inferno', maxVal=None):
    imageHalfSizeInPoints= psf.shape[0]/2
    roi= [int(imageHalfSizeInPoints*(1-ext)), int(imageHalfSizeInPoints*(1+ext))]
    psfZoom = psf[roi[0]: roi[1], roi[0]:roi[1]]
    sz = psfZoom.shape
    pixelSize = 1/oversampling
    if maxVal is None:
        maxVal = np.max(psf)
    plt.imshow(np.log10(psfZoom/maxVal), extent=
               [-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
               -sz[1]/2*pixelSize, sz[1]/2*pixelSize],
               origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
    plt.xlabel(r'$\lambda/D$')
    plt.ylabel(r'$\lambda/D$')
    cbar= plt.colorbar()
    cbar.ax.set_title('Contrast')
    plt.title(title)