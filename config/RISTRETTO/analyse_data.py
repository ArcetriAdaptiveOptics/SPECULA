import os
import glob
import specula
specula.init(-1)  # Default target device
from specula.lib.radial_profile import computeRadialProfile
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from specula.base_value import BaseValue

# hdu = fits.open('/home/matte/git/SPECULA/config/RISTRETTO/calibration/im/dm_1359modes_im.fits')
# im = hdu[1].data
# plt.figure()
# plt.plot(np.diag(im.T @ im),'-o')
# plt.grid()
# plt.xscale('log')
# plt.yscale('log')

def show_psf(psf, oversampling:int=4, title:str='', ext=0.25, vmin=-8, cmap='twilight', maxVal=None):
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
               origin='lower',cmap=cmap,vmin=vmin)
    plt.xlabel(r'$\lambda/D$')
    plt.ylabel(r'$\lambda/D$')
    cbar= plt.colorbar()
    cbar.ax.set_title('Contrast')
    plt.title(title)

# Find all directories in ./output starting with '20'
dirs = [d for d in glob.glob("./output/20*") if os.path.isdir(d)]
if not dirs:
    raise RuntimeError("No output directories found.")
# Select the most recent one (by name, assuming timestamp format)
data_dir = sorted(dirs)[-1]
print(f"Using data directory: {data_dir}")

tn = data_dir.split('/')[-1]
data = {}

# Load all .fits files in the directory
for fname in glob.glob(os.path.join(data_dir, "*.fits")):
    key = os.path.splitext(os.path.basename(fname))[0]
    with fits.open(fname) as hdul:
        arr = hdul[0].data
    data[key] = arr
    print('key:', key, 'type:', type(data[key]))

init = 200

#################### SR ######################
try:
    sr = data["sr"]
    print(f"Average Strehl Ratio after {init:1.0f} iterations: {sr[50:].mean():.4f}")
    plt.figure()
    plt.plot(sr, '-.')
    plt.title("Strehl Ratio\n"+tn)
    plt.xlabel("Frame")
    plt.ylabel("SR")
    plt.grid(True)
except FileNotFoundError:
    print(f"sr.fits file not found in {data_dir}.")

################ RESIDUALS ####################
try:
    res = data["dm_res"][init+1:, :]
    meas = data['pyr_res'][init+1:, :] 
    
    turb = data["atmo_res"][init+1:, :]

    x = np.arange(res.shape[1])+1

    # Plot RMS of residuals and turbulence
    plt.figure(figsize=(12, 6))
    plt.plot(x,np.sqrt(np.mean(turb**2, axis=0)), '-.', label='Turbulence')
    plt.plot(x,np.sqrt(np.mean(res**2, axis=0)), '-.', label='AO residuals')

    plt.plot(x[:meas.shape[1]],np.sqrt(np.mean(meas**2, axis=0)), '--',label='Measured residuals')

    plt.title("Modal RMS amplitude\n"+tn)
    plt.xlabel("Mode number")
    plt.ylabel("RMS [nm]")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
except FileNotFoundError:
    print(f"dm_res.fits, pyr_res.fits or atmo_res.fits files not found in {data_dir}.")

try:
    res = data["dm_res"][init+1:, :]
    turb = data["atmo_res"][init+1:, :]
    meas = data['pyr_res'][init+1:, :] 
    atmo_rms = np.sqrt(np.mean(turb**2, axis=0))
    res_rms = np.sqrt(np.mean(res**2, axis=0))
    Nmodes = meas.shape[1]
    corr = 1. - res_rms/atmo_rms
    p = np.polyfit(np.arange(Nmodes),corr[:Nmodes],3)
    c = np.minimum(np.max(corr),np.polyval(p,np.arange(Nmodes)))
    c = np.hstack([c,np.zeros(len(corr)-Nmodes)])

    # plt.figure()
    # plt.plot(corr1,label=r'$1^{st}$ stage (real)')
    # plt.plot(c1,'--',label=r'$1^{st}$ stage (smoothed)')
    # plt.plot(corr2,label=r'$2^{nd}$ stage (real)')
    # plt.plot(c2,'--',label=r'$2^{nd}$ stage (smoothed)')
    # plt.grid()
    # plt.xscale('log')
    # plt.legend()
    # plt.title('Modal attenuation')
    # plt.xlabel('Mode #')

    root_dir = './calibration/data/'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    obj1 = BaseValue(value=corr,target_device_idx=-1,precision=0)
    obj1.save(os.path.join(root_dir,'modal_attenuation_1st_stage.fits'),overwrite=True)
    obj1s = BaseValue(value=c,target_device_idx=-1,precision=0)
    obj1s.save(os.path.join(root_dir,'smoothed_modal_attenuation_1st_stage.fits'),overwrite=True)
except FileNotFoundError:
    print(f"dm1_res.fits, or atmo_res.fits files not found in {data_dir}.")

################### PSF ########################
try:
    psf = data["psf"]
    psf = np.sqrt(np.mean(psf[init+1:]**2,axis=0))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    show_psf(psf, title='PSF\n'+tn, cmap='inferno', ext=0.6, maxVal=np.max(psf))    
    coro_psf = data["coro_psf"]
    coro_psf = np.sqrt(np.mean(coro_psf[init+1:]**2,axis=0))
    plt.subplot(1,2,2)
    show_psf(coro_psf, title='Coronographic PSF\n'+tn, cmap='inferno', ext=0.6, maxVal=np.max(psf))
except FileNotFoundError:
    print(f"psf.fits file not found in {data_dir}.")

try:
    psf_dl = data["ref_psf"][-1]
    psf = data["psf"]
    # psf_dl = np.sqrt(np.mean(psf_dl[init+1:]**2,axis=0))
    psf = np.sqrt(np.mean(psf[init+1:]**2,axis=0))
    coro_psf = data["coro_psf_std"][0]
    oversampling = 4
    rad_psf, dist = computeRadialProfile(psf)
    rad_psf_dl, dist = computeRadialProfile(psf_dl)
    rad_cpsf, dist = computeRadialProfile(coro_psf)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(dist/oversampling, rad_psf/np.max(psf_dl), label=r'INT', c='blue')
    plt.plot(dist/oversampling, rad_psf_dl/np.max(psf_dl), '--', label='Diffraction limit', c='black')
    plt.legend()
    plt.yscale('log')
    plt.xlim([0,30])
    plt.ylim([1e-7,1])
    plt.grid()
    plt.title('PSF radial profile (RMS)\n'+tn)
    plt.xlabel(r'$\lambda/D$')
    plt.subplot(1,2,2)
    plt.plot(dist/oversampling, rad_cpsf/np.max(psf_dl), c='blue')
    plt.yscale('log')
    plt.xlim([0,30])
    plt.ylim([1e-7,1])
    plt.grid()
    plt.title('Coronographic PSF radial profile (Std Dev)\n'+tn)
    plt.xlabel(r'$\lambda/D$')
except FileNotFoundError:
    print(f"coro_psf.fits file not found in {data_dir}.")

plt.show()