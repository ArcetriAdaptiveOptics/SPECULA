import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)

import numpy as np
import os
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from specula.lib.make_mask import make_mask
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.recmat import Recmat
from specula.data_objects.m2c import M2C
from specula.calib_manager import CalibManager
from specula import cpuArray

from astropy.io import fits

def compute_and_save_influence_functions(root_dir:str, tag:str, pupil_pixels:int, n_acts:int, geom:str='circular',
                                         r0:float=10e-2, L0:float=25, zern_modes:int=2, D:float=8.0,
                                         obsratio:float=0.0, diaratio:float=1.0, doMechCoupling:bool=False,
                                         couplingCoeffs=[0.31,0.05], pupil_mask_tag=None, shrink_coords:float=1.0):
    """
    Compute zonal influence functions and modal basis for the SCAO tutorial
    Follows the same approach as test_modal_basis.py
    """
    # create calibration directory if it doesn't exist
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

    # try:
    #     kl_basis_inv = IFuncInv.restore(base_inv_filename)
    #     ifunc = IFunc.restore(ifunc_filename)
    #     m2c = M2C.restore(m2c_filename)
    #     print("Files already exist - skipping computation")
    #     return
    # except FileNotFoundError:
    #     pass


    # DM and pupil parameters for VLT-like telescope
    pupil_pixels = pupil_pixels
    n_actuators = n_acts
    telescope_diameter = D
    obsratio = obsratio 
    diaratio = diaratio

    # Actuator geometry - aligned with test_modal_basis.py
    angleOffset = 0              # No rotation

    # Actuator slaving (disable edge actuators outside pupil)
    doSlaving = True             # Enable slaving (very simple slaving)
    slavingThr = 0.1             # Threshold for master actuators
    oversampling = 4           # Minimum oversampling for FFT computations

    # Computation parameters
    dtype = specula.xp.float32   # Use current device precision

    print("Computing zonal influence functions...")
    print(f"Pupil pixels: {pupil_pixels}")
    print(f"Actuators: {n_actuators}x{n_actuators} = {n_actuators**2}")
    print(f"Telescope diameter: {telescope_diameter}m")
    print(f"Central obstruction: {obsratio*100:.1f}%")
    print(f"r0 = {r0}m, L0 = {L0}m")


    if pupil_mask_tag is not None:
        fname = os.path.join(root_dir,'pupilstop/'+pupil_mask_tag+f'_{pupil_pixels:1.0f}pixels.fits')
        hdu = fits.open(fname)
        pupil_mask = hdu[1].data
    else:
        pupil_mask = make_mask(np_size=pupil_pixels, diaratio=1.0, obsratio=obsratio)
        
    unobs_pupil_mask = make_mask(np_size=pupil_pixels, diaratio=1.0)

    # Step 1: Generate zonal influence functions
    influence_functions,mask,coords,slaveMat = compute_zonal_ifunc(
        pupil_pixels,
        n_actuators,
        geom=geom,
        angle_offset=angleOffset,
        do_mech_coupling=doMechCoupling,
        coupling_coeffs=couplingCoeffs,
        do_slaving=doSlaving,
        slaving_thr=slavingThr,
        obsratio=obsratio,
        diaratio=diaratio*shrink_coords,
        mask=specula.xp.array(pupil_mask),
        xp=specula.xp,
        dtype=dtype,
        # return_coordinates=False,
    )

    S = np.linalg.svd(influence_functions,compute_uv=False)
    fits.writeto(os.path.join(root_dir,'ifunc','eigenvalues.fits'),S)
    fits.writeto(os.path.join(root_dir,'ifunc','mask.fits'),mask)
    fits.writeto(os.path.join(root_dir,'ifunc','act_coords.fits'),coords)

    # Print statistics
    n_valid_actuators = influence_functions.shape[0]
    n_pupil_pixels = specula.xp.sum(pupil_mask)

    print(f"\nZonal influence functions:")
    print(f"Valid actuators: {n_valid_actuators}/{n_actuators**2} ({n_valid_actuators/(n_actuators**2)*100:.1f}%)")
    print(f"Pupil pixels: {int(n_pupil_pixels)}/{pupil_pixels**2} ({float(n_pupil_pixels)/(pupil_pixels**2)*100:.1f}%)")
    print(f"Influence functions shape: {influence_functions.shape}")

    # Step 2: Generate modal basis (KL modes)
    print(f"\nGenerating KL modal basis...")

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=specula.xp.array(pupil_mask), #specula.xp.array(unobs_pupil_mask),#
        diameter=telescope_diameter,
        influence_functions=influence_functions,
        r0=r0,
        L0=L0,
        zern_modes=zern_modes,
        oversampling=oversampling,
        if_max_condition_number=1e+2,
        xp=specula.xp,
        dtype=dtype
    )

    # kl_basis = remap_on_new_mask(kl_basis,(1-unobs_pupil_mask).astype(bool),(1-pupil_mask).astype(bool),specula.xp)

    print(f"KL basis shape: {kl_basis.shape}")
    print(f"Number of KL modes: {kl_basis.shape[0]}")

    kl_basis_inv = np.linalg.pinv(kl_basis)

    # Step 3: Create output directory
    os.makedirs(os.path.join(root_dir, 'ifunc'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'm2c'), exist_ok=True)

    # Step 4: Save using SPECULA data objects
    print(f"\nSaving influence functions and modal basis...")

    # fits.writeto(os.path.join(root_dir, 'ifunc', tag+'_turb_cov.fits'),cpuArray(singular_values['S2']),overwrite=True)

    # Create IFunc object and save
    ifunc_obj = IFunc(
        ifunc=influence_functions,
        mask=pupil_mask
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
        mask=pupil_mask
    )
    ifunc_inv_obj.save(base_inv_filename, overwrite=True)
    print("OK: " + base_inv_filename + " (inverse modal base)")

    # Step 5: Optional visualization
    try:
      import matplotlib.pyplot as plt

      print(f"\nGenerating visualization...")

      plt.figure(figsize=(10, 6))
      plt.semilogy(cpuArray(singular_values['S1']), 'o-', label='IF Covariance')
      plt.semilogy(cpuArray(singular_values['S2']), 'o-', label='Turbulence Covariance')
      plt.xlabel('Mode number')
      plt.ylabel('Singular value')
      plt.title('Singular values of covariance matrices')
      plt.legend()
      plt.grid(True)

      # move to CPU / numpy for plotting if required
      kl_basis = cpuArray(kl_basis)
      pupil_mask = cpuArray(pupil_mask)

      # Plot some modes
      max_modes = min(20, kl_basis.shape[0])

      # Create a mask array for display
      mode_display = np.zeros((max_modes, pupil_mask.shape[0], pupil_mask.shape[1]))

      # Place each mode vector into the 2D pupil shape
      idx_mask = np.where(pupil_mask)
      mode_ids = np.zeros(max_modes,dtype=int)
      for i in range(max_modes//2):
          mode_img = np.zeros(pupil_mask.shape)
          mode_ids[i] = i+1
          mode_img[idx_mask] = kl_basis[i]
          mode_display[i] = mode_img
      for i in range(max_modes//2,max_modes):
          mode_img = np.zeros(pupil_mask.shape)
          mode_ids[i] = kl_basis.shape[0]-max_modes+i
          mode_img[idx_mask] = kl_basis[mode_ids[i]]
          mode_display[i] = mode_img

    #   plt.figure()
    #   plt.plot(np.diag(kl_basis @ kl_basis.T),'-o')
    #   plt.grid()
    #   plt.xscale('log')
    #   plt.yscale('log')

      # Plot the reshaped modes
      n_rows = int(np.round(np.sqrt(max_modes)))
      n_cols = int(np.ceil(max_modes / n_rows))
      plt.figure(figsize=(18, 12))
      for i in range(max_modes):
          plt.subplot(n_rows, n_cols, i+1)
          plt.imshow(np.ma.masked_array(mode_display[i],mask=1-pupil_mask),origin='lower',cmap='RdBu')
          plt.title(f'Mode {mode_ids[i]}')
          plt.axis('off')
      plt.tight_layout()

      plt.show()

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    print(f"\nInfluence functions and modal basis computation completed!")
    print(f"Files saved using CalibManager in: {calib_manager.root_dir}")
    print(f"\nFiles created:")
    print(f"  ifunc/{ifunc_tag}.fits        - Zonal influence functions ({n_valid_actuators} actuators)")
    print(f"  ifunc/{base_inv_tag}.fits     - KL modal basis inverse ({kl_basis.shape[0]} modes)")
    print(f"  ifunc/{m2c_tag}.fits          - Modes-to-command base")

    # Step 6: Test loading the saved files
    print(f"\nTesting file loading...")

    try:
        # Test IFunc loading
        loaded_ifunc = IFunc.restore(ifunc_filename)
        assert loaded_ifunc.influence_function.shape == influence_functions.shape
        print("OK: IFunc loading test passed")

        # Test M2C loading
        loaded_m2c = M2C.restore(m2c_filename)
        assert loaded_m2c.m2c.shape == m2c.shape
        print("OK: M2C loading test passed")

    except Exception as e:
        print(f"⚠ File loading test failed: {e}")
    return ifunc_obj, m2c_obj


def compute_and_save_dcao_matrix(root_dir,first_stage_tag:str, second_stage_tag:str, N1_modes:int, N2_modes:int): 
    calib_manager = CalibManager(root_dir)

    base_inv_filename = calib_manager.filename('ifunc', first_stage_tag+'_kl_inv')
    MBInv = IFuncInv.restore(base_inv_filename)   
    m2s_1 = MBInv.ifunc_inv.copy() 
    base_inv_filename = calib_manager.filename('ifunc', second_stage_tag+'_kl_inv')
    MBInv = IFuncInv.restore(base_inv_filename)   
    m2s_2 = MBInv.ifunc_inv.copy() 

    m1_to_m2 = np.linalg.pinv(m2s_2[:,:N2_modes]) @ m2s_1[:,:N1_modes]

    m2m_obj = Recmat(recmat=m1_to_m2)
    os.makedirs(os.path.join(root_dir,'rec'),exist_ok=True)
    m2m_filename = calib_manager.filename('rec', first_stage_tag+f'_{N1_modes}modes_to_'+second_stage_tag+f'_{N2_modes}modes')
    m2m_obj.save(m2m_filename, overwrite=True)
    print("Saved " + m2m_filename)

    loaded_m2m = Recmat.restore(m2m_filename)
    assert loaded_m2m.recmat.shape == (N2_modes,N1_modes)
    print("OK: m2m loading test passed")


def save_m2c_as_recmat(root_dir:str, filename:str, m2c_tag):
    calib_manager = CalibManager(root_dir)  
    m2c_filename = calib_manager.filename('m2c', m2c_tag)
    loaded_m2c = M2C.restore(m2c_filename)
    rec_obj = Recmat(recmat=loaded_m2c.m2c)
    savename = os.path.join(root_dir,'rec',filename)
    rec_obj.save(savename, overwrite=True)
    print("Saved " + savename)


if __name__ == "__main__":
    xao_dir = '/raid1/mmenessini/calibration/XAO'
    soul_dir = '/raid1/mmenessini/calibration/SOUL'
    ekarus_dir = '/raid1/mmenessini/calibration/EKARUS'
    
    # Npix = 160
    # compute_and_save_influence_functions(root_dir,tag='bmc2k_vlt', pupil_pixels=Npix, n_acts=50,
    #                                       geom='alpao', r0=10e-2, obsratio=0.0, pupil_mask_tag='vlt_pupil')
    # compute_and_save_influence_functions(root_dir,tag='dm241_vlt', pupil_pixels=Npix, n_acts=17,
    #                                       geom='alpao', r0=10e-2, obsratio=0.0, pupil_mask_tag='vlt_pupil')
    # compute_and_save_dcao_matrix(root_dir,first_stage_tag='bmc2k_vlt',second_stage_tag='dm468_vlt',N1_modes=1300,N2_modes=300)

    # Npix = 160
    # compute_and_save_influence_functions(root_dir,tag='asm', pupil_pixels=Npix, n_acts=30,
    #                                       geom='circular', r0=10e-2, obsratio=0.11, D=8.4)

    # save_m2c_as_recmat(root_dir=soul_dir, m2c_tag='asm_m2c', filename='dummy_asm_m2c')

    Npix = 120
    compute_and_save_influence_functions(ekarus_dir,tag='dm468', pupil_pixels=Npix, n_acts=24,
                                          geom='alpao', r0=5e-2, pupil_mask_tag='copernico_pupil', D=1.82)
