import specula
import numpy as np


rMods = np.array([2,3,4,5,6])
n_subaps = np.array([10,20,30,40])
n_modes = np.array([54,200,450,660])
seeings = np.array([0.6,0.8,1.0,1.2,1.4])


main_config = 'soul_main.yml'

# 1. Calibrate pupdata vs n_subaps
for n_subap in n_subaps:
    pup_dist = 48/40*n_subap
    overrides = ("{"
                f"pyr.pup_diam: {n_subap:.1f}, "
                f"pyr.pup_dist: {pup_dist:.1f}, "
                f"pyr_pupdata.output_tag: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}"
                "}")
    specula.main_simul(yml_files=[main_config, 'calib_pupdata.yml'], overrides=overrides)

# 2. Calibrate sn vs n_subaps, rMods
for n_subap in n_subaps:
    pup_dist = 48/40*n_subap
    for rMod in rMods:
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}"
                    f"pyr_sn.output_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_sn"
                    "}")
        specula.main_simul(yml_files=[main_config, 'calib_sn.yml'], overrides=overrides)

# 3. Calibrate IM vs n_subaps, rMods
for n_subap in n_subaps:
    pup_dist = 48/40*n_subap
    for rMod in rMods:
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}"
                    f"pyr_im_calibrator.im_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im"
                    "}")
        specula.main_simul(yml_files=[main_config, 'calib_im.yml'], overrides=overrides)

# 3.5 Compute Rec

# 4. Calibrate aliasing vs n_subaps, n_modes, r0

# 5. Calibrate SIMPC vs n_subap, rMods, r0



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