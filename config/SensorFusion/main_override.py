import os
import specula
import numpy as np
from astropy.io import fits

from specula.mmlib.utils import get_pupil_mask, read_freq, get_psd
from specula.mmlib.compute_rec import compute_and_save_rec


rMods = np.array([0,1,3])
dotRadii = np.array([1.0,1.5,2.0])/2.0
n_subaps = np.array([12,24,36,48])
n_modes = np.array([75,300,660,1300])
# seeings = np.array([0.6,0.8,1.0,1.2,1.4])
max_pup_dist = 60

npix = 120

main_config = 'xao_main.yml'
root_dir='/raid1/mmenessini/calibration/XAO'


# 1. Calibrate pupdata vs n_subaps
for n_subap in n_subaps:
    pup_dist = np.max((16,max_pup_dist/max(n_subaps)*n_subap))
    overrides = ("{"
                f"pyr.pup_diam: {n_subap:.1f}, "
                f"pyr.pup_dist: {pup_dist:.1f}, "
                f"pyr_pupdata.output_tag: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                "}")
    try:
        specula.main_simul(yml_files=[main_config, 'calib_pupdata.yml'], overrides=overrides)
    except OSError:
        pass

# 2. Calibrate sn vs n_subaps, rMods
for n_subap in n_subaps:
    pup_dist = np.max((16,max_pup_dist/max(n_subaps)*n_subap))
    for rMod,dotRadius in zip(rMods,dotRadii):
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_sn.output_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_sn', "
                    f"zwfs.pup_diam: {n_subap:.1f}, "
                    f"zwfs.spot_radius_lambda: {dotRadius:.1f}, "
                    f"zwfs_slopes.pup_diam: {n_subap:.1f}, "
                    f"zwfs_sn.output_tag: 'z{dotRadius:1.1f}wfs_{n_subap:.0f}x{n_subap:.0f}_sn', "
                    f"data_store.store_dir:         '{root_dir}/frames/', "  
                    f"data_store.create_tn: false, "
                    f"data_store.inputs.input_list: ['pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame-cred1.out_pixels','z{dotRadius:1.1f}wfs_{n_subap:.0f}x{n_subap:.0f}_frame-ocam2.out_pixels'], "
                    "}")
        try:
            specula.main_simul(yml_files=[main_config, 'calib_sn.yml'], overrides=overrides)
        except OSError:
            pass

# 2.5 compute sensor throughput
pyr_thrp = np.zeros([len(rMods),len(n_subaps)])
zwfs_thrp = np.zeros([len(dotRadii),len(n_subaps)])
for j,n_subap in enumerate(n_subaps):
    pupdatapath = os.path.join(root_dir,f'pupils/pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}.fits')
    pyr_mask = get_pupil_mask(filepath=pupdatapath,npix=npix,pyr=True)
    zwfs_mask = get_pupil_mask(npix=npix,pupdiam=n_subap,obsratio=0.0,pyr=False)
    for i,rMod in enumerate(rMods):
        frame = fits.getdata(os.path.join(root_dir,f'frames/pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame.fits'))[0]
        thrp = np.sum(frame[pyr_mask])/np.sum(frame)
        pyr_thrp[i,j] = thrp
        fits.writeto(os.path.join(root_dir,f'slopenulls/pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_throughput.fits'), np.array([thrp]))
    for i,dotRadius in enumerate(dotRadii):
        frame = fits.getdata(os.path.join(root_dir,f'frames/z{dotRadius:1.1f}wfs_{n_subap:.0f}x{n_subap:.0f}_frame.fits'))[0]
        thrp = np.sum(frame[zwfs_mask])/np.sum(frame)
        zwfs_thrp[i,j] = thrp
        fits.writeto(os.path.join(root_dir,f'slopenulls/z{dotRadius:1.1f}wfs_{n_subap:.0f}x{n_subap:.0f}_throughput.fits'), np.array([thrp]))
print(pyr_thrp,zwfs_thrp)

# 3. Calibrate IM vs n_subaps, rMods
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((16,max_pup_dist/max(n_subaps)*n_subap))
    for rMod,dotRadius in zip(rMods,dotRadii):
        pyr_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        pyr_im_tag = pyr_tag+'_im'        
        zwfs_tag = f'z{dotRadius:1.1f}wfs_{n_subap:.0f}x{n_subap:.0f}'
        zwfs_im_tag = zwfs_tag+'_im'
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_im_calibrator.im_tag: '{pyr_im_tag}', "
                    f"zwfs.pup_diam: {n_subap:.1f}, "
                    f"zwfs.fft_res: 16.0, "
                    f"zwfs.spot_radius_lambda: {dotRadius:.1f}, "
                    f"zwfs_slopes.pup_diam: {n_subap:.1f}, "
                    f"zwfs_im_calibrator.im_tag: '{zwfs_im_tag}', "
                    "}")
        try:
            specula.main_simul(yml_files=[main_config, 'calib_im.yml'], overrides=overrides)
        except OSError:
            pass
        for N in n_modes[:i]:
            rec_tag = pyr_tag+f'_{N:1.0f}modes_rec'
            compute_and_save_rec(root_dir, im_tag=pyr_im_tag, rec_tag=rec_tag, Nmodes=N)
            rec_tag = zwfs_tag+f'_{N:1.0f}modes_rec'
            compute_and_save_rec(root_dir, im_tag=zwfs_im_tag, rec_tag=rec_tag, Nmodes=N)


# # 4. Calibrate aliasing vs n_subaps, n_modes, r0
# aliaspath = os.path.join(root_dir,'aliasing')
# os.makedirs(aliaspath,exist_ok=True)
# fs = read_freq(params_path=f'./{main_config}')
# for i,n_subap in enumerate(n_subaps):
#     pup_dist = 48/40*n_subap
#     for rMod in rMods:
#         for seeing in seeings:
#             for N in n_modes[:i]:
#                 overrides = ("{"
#                             f"pyr.pup_diam: {n_subap:.1f}, "
#                             f"pyr.pup_dist: {pup_dist:.1f}, "
#                             f"pyr.mod_amp: {rMod:.1f}, "
#                             f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}, "
#                             f"seeing.constant: {seeing:1.1f}, "
#                             f"perfect_dm.nmodes: {N:1.0f}, "
#                             "}")
#                 specula.main_simul(yml_files=[main_config, 'calib_aliasing.yml'], overrides=overrides)
#                 alias_modes = fits.getdata(os.path.join(root_dir,'scratch_aliasing','pyr_modes.fits'))
#                 psd,f = get_psd(alias_modes.T, nperseg=1024, dt=1/fs)
#                 tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}_{N:1.0f}modes_alias_PSD'
#                 fits.writeto(os.path.join(aliaspath,tag+'.fits'),psd)
#                 print('Saved aliasing PSD as: '+tag)


# # 4.5 Compute correction vectors for SIMPC
# # ...

# # 5. Calibrate SIMPC vs n_subap, rMods, r0/correction
# ogpath = os.path.join(root_dir,'optgains')
# os.makedirs(ogpath,exist_ok=True)
# for i,n_subap in enumerate(n_subaps):
#     pup_dist = 48/40*n_subap
#     for rMod in rMods:
#         im_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im'
#         im = fits.getdata(os.path.join(root_dir,'im',im_tag+'.fits'))
#         im_norm = np.diag(im.T @ im)
#         for seeing in seeings:
#             tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}'
#             simpc_tag = tag+'_simpc'
#             overrides = ("{"
#                         f"pyr.pup_diam: {n_subap:.1f}, "
#                         f"pyr.pup_dist: {pup_dist:.1f}, "
#                         f"pyr.mod_amp: {rMod:.1f}, "
#                         f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}, "
#                         f"seeing_random.constant: {seeing:1.1f}, "
#                         f"pyr_im_calibrator.im_tag: '{im_tag}', "
#                         "}")
#             specula.main_simul(yml_files=[main_config, 'calib_simpc.yml'], overrides=overrides)

#             simpc = im = fits.getdata(os.path.join(root_dir,'im',simpc_tag+'.fits'))
#             og = np.diag(simpc.T @ im)/im_norm
#             fits.writeto(os.path.join(ogpath,tag+'_og.fits'))
#             print('Saved optical gains as: '+tag+'_og')
#             # for N in n_modes[:i]:
#             #     rec_tag = tag+f'_{N:1.0f}modes_rec'
#             #     compute_and_save_rec(root_dir, im_tag=im_tag, rec_tag=rec_tag, Nmodes=N)



# # Range of gains to test
# gains = np.linspace(0.1, 1.0, 10) #(0.2, 0.9, 8)#
# output_dir = "gain_override"
# base_config = "xao_main.yml"

# for gain in gains:
#     overrides = ("{"
#                 "main.total_time: 0.5, "
#                 f"filter.iir_gain: {gain:.2f}, "
#                 # f"filter.g_track: {gain:.2f}, "
#                 f"data_store.store_dir: ./output/gain_opt/gain_{gain:.2f}"
#                 "}")

#     specula.main_simul(yml_files=[base_config], overrides=overrides)