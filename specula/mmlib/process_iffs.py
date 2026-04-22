import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)

import numpy as np
import os

from scipy.io import readsav

from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.m2c import M2C
from specula.calib_manager import CalibManager

def postprocess_iffs(root_dir:str, data_path:str, tag:str, D:float):
    fname = os.path.join(data_path,'phase_matrix.sav')
    data_dict = readsav(fname)
    
    kl_basis = data_dict['klmatrix'].T
    dpix = data_dict['dpix']
    m2c_full = data_dict['klm2c']
    m2c = m2c_full[:,:kl_basis.shape[0]]
    idx = data_dict['idx_mask']

    print('Reading data from: '+fname)
    print(f'Scale: {dpix} pixels across diameter')

    # Pupil mask
    pmask = np.ones([dpix,dpix],dtype=bool).flatten()
    pmask[idx] = False
    pmask = pmask.reshape([dpix,dpix])
    simul_params = SimulParams(pixel_pupil=dpix,pixel_pitch=D/dpix)
    pupil_mask = Pupilstop(simul_params=simul_params, input_mask=pmask)

    # Influence functions
    influence_functions = kl_basis.T @ specula.xp.linalg.pinv(m2c)

    mask_pixels = np.sum(1-pmask)
    pupil_pixels = influence_functions.shape[0]
    print(f"Valid mask pixels: {mask_pixels}")
    print(f"Pupil pixels: {pupil_pixels}")

    ##########################################################    
    os.makedirs(root_dir, exist_ok=True)

    # initialize calibration manager
    calib_manager = CalibManager(root_dir)

    # tags
    ifunc_tag = tag+'_ifunc'
    m2c_tag = tag+'_m2c'
    base_inv_tag = tag+'_kl_inv'

    ifunc_filename = calib_manager.filename('ifunc', ifunc_tag)
    m2c_filename = calib_manager.filename('m2c', m2c_tag)
    base_inv_filename = calib_manager.filename('ifunc', base_inv_tag)


    print(f"Influence functions shape: {influence_functions.shape}")
    print(f"KL basis shape: {kl_basis.shape}")
    print(f"Number of KL modes: {kl_basis.shape[0]}")

    kl_basis_inv = np.linalg.pinv(kl_basis)

    # Step 3: Create output directory
    os.makedirs(os.path.join(root_dir, 'pupilstop'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'ifunc'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'm2c'), exist_ok=True)

    # Step 4: Save using SPECULA data objects
    print(f"\nSaving influence functions and modal basis...")

    # Create IFunc object and save
    ifunc_obj = IFunc(
        ifunc=influence_functions.T,
        mask=(1-pmask).astype(np.uint8)
    )
    ifunc_obj.save(ifunc_filename, overwrite=True)
    print("OK: " + ifunc_filename + " (zonal influence functions)")

    # Create M2C object for mode-to-command matrix and save
    m2c_obj = M2C(
        m2c=m2c
    )
    m2c_obj.save(m2c_filename, overwrite=True)
    print("OK: " + m2c_filename + " (KL modal basis)")

    # inverse influence function object for modal analysis
    print(f"\nSaving inverse modal base...")
    ifunc_inv_obj = IFuncInv(
        ifunc_inv=kl_basis_inv,
        mask=(1-pmask).astype(np.uint8)
    )
    ifunc_inv_obj.save(base_inv_filename, overwrite=True)
    print("OK: " + base_inv_filename + " (inverse modal base)")

    fname = os.path.join(root_dir, 'pupilstop', tag+f'_{dpix:1.0f}pixels.fits')
    pupil_mask.save(fname)






if __name__ == "__main__":
    D = 8.4
    data_path = '/raid1/mmenessini/LBTData/KLv30dx'
    soul_dir = '/raid1/mmenessini/calibration/SOUL/KLv30dx'

    postprocess_iffs(root_dir=soul_dir, data_path=data_path, tag='asm_v30dx', D=D)



