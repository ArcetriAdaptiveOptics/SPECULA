import os
import specula
specula.init(0)

import numpy as np
from astropy.io import fits

from specula.mmlib.yaml_overrides import write_yaml_overrides
from specula.mmlib.utils import get_pupil_mask, read_freq, r0_from_seeing
from specula.mmlib.compute_rec import compute_and_save_rec

# from specula.mmlib.save_correction_vector import save_correction_vector


rMods = np.array([3,4,5,6])
n_subaps = np.array([10,20,30,40])
n_modes = np.array([54,120,400])
seeings = np.array([1.8,2.0,2.2,2.4,2.6])

max_pup_dist = 60
min_pup_dist = 14

npix = 120

main_config = 'ekarus_main.yml'
root_dir='/raid1/mmenessini/calibration/EKARUS'


# # 1. Calibrate pupdata vs n_subaps
# for n_subap in n_subaps:
#     pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
#     overrides = ("{"
#                 f"pyr.pup_diam: {n_subap:.1f}, "
#                 f"pyr.pup_dist: {pup_dist:.1f}, "
#                 f"pyr_pupdata.output_tag: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
#                 "}")
#     write_yaml_overrides(input_string=overrides)
#     try:
#         os.system(f"specula {main_config} calib_pupdata.yml temp_overrides.yml")
#         # specula.main_simul(yml_files=[main_config, 'calib_pupdata.yml'], overrides=overrides)
#     except FileExistsError: #OSError:
#         pass

# # 2. Calibrate sn vs n_subaps, rMods
# for n_subap in n_subaps:
#     pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
#     for rMod in rMods:
#         overrides = ("{"
#                     f"pyr.pup_diam: {n_subap:.1f}, "
#                     f"pyr.pup_dist: {pup_dist:.1f}, "
#                     f"pyr.mod_amp: {rMod:.1f}, "
#                     f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
#                     f"pyr_sn.output_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_sn', "
#                     f"data_store.store_dir:         '{os.path.join(root_dir,'frames')}', "  
#                     f"data_store.create_tn: false, "
#                     f"data_store.inputs.input_list: ['pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame_null-ocam.out_pixels'], "
#                     "}")
#         write_yaml_overrides(input_string=overrides)
#         try:
#             os.system(f"specula {main_config} calib_sn.yml temp_overrides.yml")
#             # specula.main_simul(yml_files=[main_config, 'calib_sn.yml'], overrides=overrides)
#         except FileExistsError: #OSError:
#             pass

# # 2.5 compute sensor throughput
# pyr_thrp = np.zeros([len(rMods),len(n_subaps)])
# for j,n_subap in enumerate(n_subaps):
#     pupdatapath = os.path.join(root_dir,f'pupils/pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}.fits')
#     pyr_mask = get_pupil_mask(filepath=pupdatapath,npix=npix,pyr=True)
#     for i,rMod in enumerate(rMods):
#         frame = fits.getdata(os.path.join(root_dir,f'frames/pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame_null.fits'))[0]
#         thrp = np.sum(frame[pyr_mask])/np.sum(frame)
#         pyr_thrp[i,j] = thrp
#         fits.writeto(os.path.join(root_dir,f'slopenulls/pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_throughput.fits'), np.array([thrp]), overwrite=True)
# print(pyr_thrp)

# 3. Calibrate IM vs n_subaps, rMods
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        im_tag = tag+'_im'
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_im_calibrator.im_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im', "
                    "}")
        write_yaml_overrides(input_string=overrides)
        try:
            os.system(f"specula {main_config} calib_im.yml temp_overrides.yml")
            # specula.main_simul(yml_files=[main_config, 'calib_im.yml'], overrides=overrides)
        except FileExistsError: #OSError:
            pass

for i,n_subap in enumerate(n_subaps):
    for rMod in rMods:
        tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        im_tag = tag+'_im'
        for N in n_modes[:i+1]:
            rec_tag = tag+f'_{N:1.0f}modes'
            compute_and_save_rec(root_dir, im_tag=im_tag, rec_tag=rec_tag, Nmodes=N, overwrite=True)

# # 3.25 compute ML reconstructors
# for i,n_subap in enumerate(n_subaps):
#     pupdatapath = os.path.join(root_dir,f'pupils/pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}.fits')
#     pyr_mask = get_pupil_mask(filepath=pupdatapath,npix=npix,pyr=True)
#     for rMod in rMods:
#         frame = fits.getdata(os.path.join(root_dir,f'frames/pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_frame_null.fits'))[0]
#         tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
#         im_tag = tag+'_im'
#         for N in n_modes[:i+1]:
#             for seeing in seeings:
#                 r0 = r0_from_seeing(seeing)
#                 sn = frame[pyr_mask]
#                 rec_tag = tag+f'_s{seeing:1.1f}_{N:1.0f}modes_mmse'
#                 compute_and_save_rec(root_dir, im_tag=im_tag, mmse=True, RON=0.5, slope_null=sn, r0=r0,
#                                     rec_tag=rec_tag, Nmodes=N, overwrite=True, diam=1.82)

# 3.5 Compute correction vectors for SIMPC
os.makedirs(os.path.join(root_dir,'scratch_corrvec'),exist_ok=True)
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    N = n_modes[min(i,len(n_modes)-1)]
    rec_tag = f'pyr5.0_{n_subap:.0f}x{n_subap:.0f}_{N:1.0f}modes_rec'
    for seeing in seeings:
        # rec_tag = f'pyr5.0_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}_{N:1.0f}modes_rec_mmse'
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: 5.0, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_slopes.sn_object:  'pyr5.0_{n_subap:.0f}x{n_subap:.0f}_sn', "
                    f"pyr_modalrec.recmat_object: {rec_tag}, "
                    f"filter.n_modes:   [{N:1.0f}], "
                    f"dm.nmodes:   {N:1.0f}, "
                    f"seeing.constant: {seeing:1.1f}, "
                    f"data_store.store_dir:         '{os.path.join(root_dir,'scratch_corrvec')}', "
                    f"data_store.inputs.input_list: ['s{seeing:1.1f}_{N:1.0f}modes_atmo-atmo_modes.out_modes','s{seeing:1.1f}_{N:1.0f}modes_res-dm_mode_res.out_modes'], "
                    "}")
        write_yaml_overrides(input_string=overrides)
        tag = f's{seeing:1.1f}_{N:1.0f}modes_corrvec'
        try:
            corrvec = fits.getdata(os.path.join(root_dir,'scratch_corrvec',tag+'.fits'))
            print('Correction vector '+tag+' already exists: skipping computation')
        except FileNotFoundError:
            os.system(f"specula {main_config} calib_corrvec.yml temp_overrides.yml")
            atmo_modes = fits.getdata(os.path.join(root_dir,'scratch_corrvec',f's{seeing:1.1f}_{N:1.0f}modes_atmo.fits'))
            res_modes = fits.getdata(os.path.join(root_dir,'scratch_corrvec',f's{seeing:1.1f}_{N:1.0f}modes_res.fits'))
            atmo_rms = np.sqrt(np.mean(atmo_modes**2,axis=0))
            res_rms = np.sqrt(np.mean(res_modes**2,axis=0))
            corrvec = 1.0-res_rms/atmo_rms
            corrvec = np.maximum(0.0,corrvec)
            fits.writeto(os.path.join(root_dir,'scratch_corrvec',tag+'.fits'),corrvec)
            print('Saved correction vector as: '+tag)

# Add perfect correction vectors (seeing 0")
atmo_modes = fits.getdata(os.path.join(root_dir,'scratch_corrvec',f's{seeing:1.1f}_{N:1.0f}modes_atmo.fits'))
L = len(atmo_modes)
for i,N in enumerate(n_modes):
    tag = f's0.0_{N:1.0f}modes_corrvec'
    try:
        corrvec = fits.getdata(os.path.join(root_dir,'data',tag+'.fits'))
        print('Correction vector '+tag+' already exists: skipping computation')
    except FileNotFoundError:
        corrvec = np.zeros(L)
        corrvec[:N] = 1.0 # perfect correction up to mode N
        fits.writeto(os.path.join(root_dir,'data',tag+'.fits'),corrvec)
        print('Saved correction vector as: '+tag)


# 4. Calibrate SIMPC vs n_subap, rMods, r0/correction
ogpath = os.path.join(root_dir,'optgains')
os.makedirs(ogpath,exist_ok=True)
fs = read_freq(params_path=f'./{main_config}')
ncycles = 40
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    N = n_modes[min(i,len(n_modes)-1)]
    for rMod in rMods:
        im_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im'
        im = fits.getdata(os.path.join(root_dir,'im',im_tag+'.fits'))
        im = im[:,:N]
        im_norm = np.diag(im.T @ im)
        for seeing in seeings:
            cv_tag = f's{seeing:1.1f}_{N:1.0f}modes_corrvec'
            tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}'
            simpc_tag = tag+'_simpc'
            overrides = ("{"
                        f"main.total_time: {N*2*ncycles/fs}, "
                        f"atmo_random.update_interval: {N*2:1.0f}, "
                        f"pyr.pup_diam: {n_subap:.1f}, "
                        f"pyr.pup_dist: {pup_dist:.1f}, "
                        f"pyr.mod_amp: {rMod:.1f}, "
                        f"pushpull.nmodes: {N:1.0f}, "
                        f"pushpull.ncycles: {ncycles:1.0f}, "
                        f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                        f"dm_random.nmodes: {N:1.0f}, "
                        f"dm.nmodes: {N:1.0f}, "
                        f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                        f"seeing_random.constant: {seeing:1.1f}, "
                        f"scale_random.constant_mul_data: {cv_tag}, "
                        f"pyr_im_calibrator.im_tag: '{simpc_tag}', "
                        f"data_store.store_dir:         '{os.path.join(root_dir,'scratch_simpc')}', "  
                        f"data_store.create_tn: false, "
                        f"data_store.inputs.input_list: ['s{seeing:1.1f}_{N:1.0f}modes_atmo-atmo_pc_modes.out_modes'], " #,'{N:1.0f}modes_pushpull-pushpull.output'
                        "}")
            write_yaml_overrides(input_string=overrides)
            try:
                os.system(f"specula {main_config} calib_simpc.yml temp_overrides.yml")
                # specula.main_simul(yml_files=[main_config, 'calib_simpc.yml'], overrides=overrides)
                simpc = fits.getdata(os.path.join(root_dir,'im',simpc_tag+'.fits'))
                og = np.diag(simpc.T @ im)/im_norm
                cog = np.sqrt(np.diag(simpc.T @ simpc)/im_norm - og**2)
                atmo_modes = fits.getdata(os.path.join(root_dir,'scratch_simpc',f's{seeing:1.1f}_{N:1.0f}modes_atmo.fits'))
                atmo_rms = np.sqrt(np.mean(atmo_modes**2,axis=0))
                atmo_res = np.sqrt(np.sum(atmo_rms[:N]**2))
                tag += f'_{atmo_res:1.0f}Nm'
                fits.writeto(os.path.join(ogpath,tag+'_og.fits'),og)
                print('Saved optical gains as: '+tag+'_og')
                fits.writeto(os.path.join(ogpath,tag+'_compl_og.fits'),cog)
                print('Saved complementary (perpedicular) optical gains as: '+tag+'_compl_og')
            except OSError:
                pass

# 4.5 Calibrate SIMPC vs n_subap, rMods for PERFECT correction
ogpath = os.path.join(root_dir,'optgains')
os.makedirs(ogpath,exist_ok=True)
fs = read_freq(params_path=f'./{main_config}')
ncycles = 40
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    N = n_modes[min(i,len(n_modes)-1)]
    cv_tag = f's0.0_{N:1.0f}modes_corrvec'
    for rMod in rMods:
        im_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im'
        im = fits.getdata(os.path.join(root_dir,'im',im_tag+'.fits'))
        im = im[:,:N]
        im_norm = np.diag(im.T @ im)
        for seeing in seeings:
            tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}'
            simpc_tag = tag+'_simpc_pl'
            overrides = ("{"
                        f"main.total_time: {N*2*ncycles/fs}, "
                        f"atmo_random.update_interval: {N*2:1.0f}, "
                        f"pyr.pup_diam: {n_subap:.1f}, "
                        f"pyr.pup_dist: {pup_dist:.1f}, "
                        f"pyr.mod_amp: {rMod:.1f}, "
                        f"pushpull.nmodes: {N:1.0f}, "
                        f"pushpull.ncycles: {ncycles:1.0f}, "
                        f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                        f"dm_random.nmodes: {N:1.0f}, "
                        f"dm.nmodes: {N:1.0f}, "
                        f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                        f"seeing_random.constant: {seeing:1.1f}, "
                        f"scale_random.constant_mul_data: {cv_tag}, "
                        f"pyr_im_calibrator.im_tag: '{simpc_tag}', "
                        f"data_store.store_dir:         '{os.path.join(root_dir,'scratch_simpc')}', "  
                        f"data_store.create_tn: false, "
                        f"data_store.inputs.input_list: ['{N:1.0f}modes_pushpull-pushpull.output'], "
                        "}")
            # write_yaml_overrides(input_string=overrides)
            # try:
            #     og = fits.getdata(os.path.join(ogpath,tag+'_og_pl.fits'))
            #     cog = fits.getdata(os.path.join(ogpath,tag+'_compl_og_pl.fits'))
            #     print('Optical gains '+tag+'_og_pl already found, skipping computation')
            # except FileNotFoundError:
            try:
                os.system(f"specula {main_config} calib_simpc.yml temp_overrides.yml")
                simpc = fits.getdata(os.path.join(root_dir,'im',simpc_tag+'.fits'))
                og = np.diag(simpc.T @ im)/im_norm
                cog = np.sqrt(np.diag(simpc.T @ simpc)/im_norm - og**2)
                fits.writeto(os.path.join(ogpath,tag+'_og_pl.fits'),og)
                print('Saved optical gains as: '+tag+'_og_pl')
                fits.writeto(os.path.join(ogpath,tag+'_compl_og_pl.fits'),cog)
                print('Saved complementary (perpedicular) optical gains as: '+tag+'_compl_og_pl')
            except OSError:
                pass

# 5. Calibrate aliasing vs n_subaps, n_modes, r0
aliaspath = os.path.join(root_dir,'aliasing')
os.makedirs(aliaspath,exist_ok=True)
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        for seeing in seeings:
            modes_vec = n_modes.copy() if i == len(n_subaps)-1 and seeing == 1.0 else np.array([n_modes[min(i,len(n_modes)-1)]])
            for N in modes_vec:
                overrides = ("{"
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_{N:1.0f}modes_rec', "
                            # f"pyr_pc_modalrec.recmat_object: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}_{N:1.0f}modes_rec', "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing.constant: {seeing:1.1f}, "
                            f"perfect_dm.nmodes: {N:1.0f}, "
                            "}")
                write_yaml_overrides(input_string=overrides)
                tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}_{N:1.0f}modes_alias'
                try:
                    alias_rms = fits.getdata(os.path.join(aliaspath,tag+'.fits'))
                    print('Aliasing power file '+tag+' already exists: skipping computation')
                except FileNotFoundError:
                    os.system(f"specula {main_config} calib_aliasing.yml temp_overrides.yml")
                    alias_modes = fits.getdata(os.path.join(root_dir,'scratch_aliasing','pyr_modes.fits'))
                    alias_rms = np.sqrt(np.mean(alias_modes**2,axis=0)) 
                    fits.writeto(os.path.join(aliaspath,tag+'.fits'),alias_rms,overwrite=True)
                    # alias_pc_modes = fits.getdata(os.path.join(root_dir,'scratch_aliasing','pyr_pc_modes.fits'))
                    # alias_pc_rms = np.sqrt(np.mean(alias_pc_modes**2,axis=0)) 
                    # fits.writeto(os.path.join(aliaspath,tag+'_pc.fits'),alias_pc_rms,overwrite=True)
                    print('Saved aliasing power as: '+tag)



