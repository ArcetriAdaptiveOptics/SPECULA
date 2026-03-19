import os
import glob
import yaml
import specula
specula.init(-1)  # Default target device

from specula.lib.calc_psf import calc_psf
from specula.lib.radial_profile import computeRadialProfile

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

from specula.base_value import BaseValue
from specula.data_objects.psd import PSD
from specula.data_objects.iir_filter_data import IirFilterData

def get_psd(data, dt:float):
    psd_obj = PSD(data,dt)
    psd = psd_obj.psd_data.copy()
    f = psd_obj.freq_vec.copy()
    return psd,f

def get_reference_psf(pupil_tag:str='vlt_pupil_160pixels',nd:int=4):
    hdu = fits.open(os.path.join('./calibration/pupilstop',pupil_tag+'.fits'))
    pupil = hdu[1].data
    psf = calc_psf(np.zeros_like(pupil),pupil.astype(float), imwidth=int(pupil.shape[1]*nd), normalize=True,
                                            xp=np, complex_dtype=np.complex128)
    return psf


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

def get_control_data(controller_name:str):
    delay_frames = 1.0 + float(params[controller_name]['delay'])
    if params[controller_name]['class'] == 'IirFilter':
        gain = float(params[controller_name]['iir_gain'])
        iir_path = os.path.join('./calibration/filter/',str(params[controller_name]['iir_filter_data_object'])+'.fits')
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

################### Parameters #########################
params_path = os.path.join(data_dir,'params.yml')

with open(params_path, 'r') as file:
    params = yaml.safe_load(file)
    fs = 1.0/float(params['main']['time_step'])
    try:
        filter_data_complex, delay_frames = get_control_data('filter')
    except:
        filter_data1, delay_frames1 = get_control_data('filter1')
        filter_data2, delay_frames2 = get_control_data('filter2')
        fs1 = 1.0/float(params['cred1']['dt'])
        fs2 = 1.0/float(params['cred2']['dt'])
        fs = np.max((fs1,fs2))
        print(fs)

init = int(0.1*fs)

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
except KeyError:
    try:
        sr1 = data["sr1"]
        sr2 = data["sr2"]
        print(f"Average Strehl Ratio after {init:1.0f} iterations: {sr2[50:].mean():.4f}")
        plt.figure()
        plt.plot(sr1, '-.',label=r'$1^{st}$ stage')
        plt.plot(sr2, '-.',label=r'$2^{nd}$ stage')
        plt.title("Strehl Ratio\n"+tn)
        plt.xlabel("Frame")
        plt.ylabel("SR")
        plt.grid(True)
    except KeyError:
        print(f"sr.fits file not found in {data_dir}.")

################ RESIDUALS ####################
try:
    res = data["dm_res"][init+1:, :]
    turb = data["atmo_res"][init+1:, :]

    x = np.arange(res.shape[1])+1
    turb_rms = np.sqrt(np.mean(turb**2, axis=0))
    res_rms = np.sqrt(np.mean(res**2, axis=0))

    # Plot RMS of residuals and turbulence
    plt.figure(figsize=(12, 6))
    plt.plot(x,turb_rms, '-.', label='Turbulence')
    plt.plot(x,res_rms, '-.', label='AO residuals')

    corr = res_rms/turb_rms
    root_dir = './calibration/'
    dir_path = os.path.join(root_dir, 'data')
    os.makedirs(dir_path, exist_ok=True)
    fname = os.path.join(dir_path,f'correction_vector_{tn}.fits')
    bv = BaseValue(description='correction_level',value=corr)
    bv.save(filename=fname,overwrite=True)
    rec_corr = bv.restore(fname)

    # plt.plot(x[:meas.shape[1]],np.sqrt(np.mean(meas**2, axis=0)), '--',label='Measured residuals')

    plt.title("Modal RMS amplitude\n"+tn)
    plt.xlabel("Mode number")
    plt.ylabel("RMS [nm]")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
except KeyError:
    try:
        res1 = data["dm1_res"][init+1:, :]
        res2 = data["dm2_res"][init+1:, :]
        turb = data["atmo_res"][init+1:, :]

        x = np.arange(turb.shape[1])+1
        turb_rms = np.sqrt(np.mean(turb**2, axis=0))
        res1_rms = np.sqrt(np.mean(res1**2, axis=0))
        res2_rms = np.sqrt(np.mean(res2**2, axis=0))

        # Plot RMS of residuals and turbulence
        plt.figure(figsize=(12, 6))
        plt.plot(x,turb_rms, '-.', label='Turbulence')
        plt.plot(x,res1_rms, '-.', label=r'$1^{st}$ stage residuals')
        plt.plot(x,res2_rms, '-.', label=r'$2^{nd}$ stage residuals')

        plt.title("Modal RMS amplitude\n"+tn)
        plt.xlabel("Mode number")
        plt.ylabel("RMS [nm]")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
    except KeyError:
        print(f"dm_res.fits, pyr_res.fits or atmo_res.fits files not found in {data_dir}.")

try:
    comm = data["dm_cmd"][init+1:, :]
    res = data["dm_res"][init+1:, :comm.shape[1]]
    meas = data["pyr_modes"][init+1:, :comm.shape[1]]
    zmeas = data["zwfs_modes"][init+1:, :comm.shape[1]]
    
    pol_modes = comm + meas
    zpol_modes = comm + zmeas
    turb_modes = res + comm

    dt = 1/fs
    pol_psd, f = get_psd(turb_modes.T,dt=dt)#,interval=interval)
    res_psd, f = get_psd(res.T,dt=dt)#,interval=interval)

    flims = [fs/init,1/dt/2]
    freq = np.logspace(-2,np.log10(fs/2),2000)
    nw_delay, dw_delay = filter_data_complex.discrete_delay_tf(delay_frames)

    lo_mode_ids = [0,1,2,3,20]
    plt.figure()
    plt.subplot(2,2,1)
    for k,mode in enumerate(lo_mode_ids):
        rtf = filter_data_complex.RTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        plt.loglog(f,pol_psd[mode,:]/np.min(pol_psd[mode,:][f<flims[-1]]),c=f'C{k}',label=f'Mode {mode:1.0f}')
        plt.loglog(freq,rtf**-2,'--',c=f'C{k}',label='')
    plt.grid(which='both', alpha=0.3)
    # plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.xlim(flims)
    plt.ylabel(r'RMS [$nm^2$]')
    plt.title('Pseudo-open-loop PSD')
    plt.subplot(2,2,3)
    for mode in lo_mode_ids:
        plt.loglog(f,res_psd[mode,:],label=f'Mode {mode:1.0f}')
    plt.grid(which='both', alpha=0.3)
    plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.xlim(flims)
    plt.ylabel(r'RMS [$nm^2$]')
    plt.title('Residuals PSD')

    ho_mode_ids = [50,100,200,500,1000]
    plt.subplot(2,2,2)
    for k,mode in enumerate(ho_mode_ids):
        rtf = filter_data_complex.RTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        plt.loglog(f,pol_psd[mode,:]/np.min(pol_psd[mode,:][f<flims[-1]]),c=f'C{k}',label=f'Mode {mode:1.0f}')
        plt.loglog(freq,rtf**-2,'--',c=f'C{k}',label='')
    plt.grid(which='both', alpha=0.3)
    # plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.xlim(flims)
    plt.ylabel(r'RMS [$nm^2$]')
    plt.title('Pseudo-open-loop PSD')
    plt.subplot(2,2,4)
    for mode in ho_mode_ids:
        plt.loglog(f,res_psd[mode,:],label=f'Mode {mode:1.0f}')
    plt.grid(which='both', alpha=0.3)
    plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.xlim(flims)
    plt.ylabel(r'RMS [$nm^2$]')
    plt.title('Residuals PSD')
    plt.tight_layout()

    # plt.figure()
    # for k,mode in enumerate(ho_mode_ids):
    #     plt.loglog(f,pol_spe[mode,:]-zpol_spe[mode,:],'--',c=f'C{k}',label=f'Mode {mode:1.0f}')
    #     # plt.loglog(f,zpol_spe[mode,:],':',c=f'C{k}',label=f'')
    # plt.grid(which='both', alpha=0.3)
    # plt.xlabel('Frequency [Hz]')
    # plt.legend()

except KeyError:
    print(f"dm_res.fits, pyr_res.fits or atmo_res.fits files not found in {data_dir}.")

################### PSF ########################
try:
    psf = data["psf"]
    psf = np.sqrt(np.mean(psf[init+1:]**2,axis=0))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    show_psf(psf, title='PSF\n'+tn, cmap='inferno', ext=0.55, vmin=-6)    
    coro_psf = data["coro_psf"]
    coro_psf = np.sqrt(np.mean(coro_psf[init+1:]**2,axis=0))
    plt.subplot(1,2,2)
    show_psf(coro_psf, title='Coronagraphic PSF\n'+tn, cmap='inferno', ext=0.55,  vmin=-6)
except KeyError:
    try:
        psf1 = data["psf1"]
        psf1 = np.sqrt(np.mean(psf1[init+1:]**2,axis=0))
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        show_psf(psf1, title=r'$1^{st}$ stage PSF\n'+tn, cmap='Reds', ext=0.55, vmin=-6)    
        psf2 = data["psf2"]
        psf2 = np.sqrt(np.mean(psf2[init+1:]**2,axis=0))
        plt.subplot(1,2,2)
        show_psf(psf2, title=r'2^{st} stage PSF\n'+tn, cmap='Blues', ext=0.55, vmin=-6)   
    except KeyError:
        print(f"psf.fits file not found in {data_dir}.")


##################### Modes ##########################
try:
    res = data['dm_res'][init+1:, :]
    pywfs_modes = data['pyr_modes'][init+1:, :]
    zwfs_modes = data['zwfs_modes'][init+1:, :]
    Nmodes = pywfs_modes.shape[1]
    x = np.arange(Nmodes)+1
    pyr_rec_rms = np.sqrt(np.mean((pywfs_modes-res[:,:Nmodes])**2,axis=0))
    zwfs_rec_rms = np.sqrt(np.mean((zwfs_modes-res[:,:Nmodes])**2,axis=0))
    plt.figure()
    plt.plot(x, pyr_rec_rms, label='pyWFS')
    plt.plot(x, zwfs_rec_rms, label='zWFS')
    plt.title('Rec error temporal RMS')
    plt.xlabel('KL mode #')
    plt.ylabel('RMS [nm]')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
except:
    print(f"pyr_modes.fits or zwfs_modes.fits file(s) not found in {data_dir}.")



################# PSF profiles #######################
oversampling = 4
psf_dl = get_reference_psf(pupil_tag='vlt_pupil_160pixels',nd=oversampling)
rad_psf_dl, dist = computeRadialProfile(psf_dl)
try:
    psf = data["psf"]
    psf = np.sqrt(np.mean(psf[init+1:]**2,axis=0))
    coro_psf = data["coro_psf_std"]
    coro_psf = np.sqrt(np.mean(coro_psf[init+1:]**2,axis=0))
    rad_psf, dist = computeRadialProfile(psf)
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
except KeyError:
    try:
        psf1 = data["psf1"]
        psf1 = np.sqrt(np.mean(psf1[init+1:]**2,axis=0))
        coro_psf1 = data["coro_psf1_std"]
        coro_psf1 = np.sqrt(np.mean(coro_psf1[init+1:]**2,axis=0))
        rad_psf1, dist = computeRadialProfile(psf1)
        rad_cpsf1, dist = computeRadialProfile(coro_psf1)
        psf2 = data["psf2"]
        psf2 = np.sqrt(np.mean(psf2[init+1:]**2,axis=0))
        coro_psf2 = data["coro_psf2_std"]
        coro_psf2 = np.sqrt(np.mean(coro_psf2[init+1:]**2,axis=0))
        rad_psf2, dist = computeRadialProfile(psf2)
        rad_cpsf2, dist = computeRadialProfile(coro_psf2)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(dist/oversampling, rad_psf1/np.max(psf_dl), label=r'$1^{st}$ stage')
        plt.plot(dist/oversampling, rad_psf2/np.max(psf_dl), label=r'$2^{nd}$ stage')
        plt.plot(dist/oversampling, rad_psf_dl/np.max(psf_dl), '--', label='Diffraction limit')
        plt.legend()
        plt.yscale('log')
        plt.xlim([0,30])
        plt.ylim([1e-7,1])
        plt.grid()
        plt.title('PSF radial profile (RMS)\n'+tn)
        plt.xlabel(r'$\lambda/D$')
        plt.subplot(1,2,2)
        plt.plot(dist/oversampling, rad_cpsf1/np.max(psf_dl), label=r'$1^{st}$ stage')
        plt.plot(dist/oversampling, rad_cpsf2/np.max(psf_dl), label=r'$2^{nd}$ stage')
        plt.yscale('log')
        plt.xlim([0,30])
        plt.ylim([1e-7,1])
        plt.grid()
        plt.title('Coronographic PSF radial profile (Std Dev)\n'+tn)
        plt.xlabel(r'$\lambda/D$')
    except KeyError:
        print(f"coro_psf.fits file not found in {data_dir}.")

plt.show()