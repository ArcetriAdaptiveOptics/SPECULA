import specula
specula.init(0)  # Default target device

import unittest

from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from test.specula_testlib import cpu_and_gpu

class TestGenerateModalBasis(unittest.TestCase):

    @cpu_and_gpu
    def test_influence_functions_and_mask(self, target_device_idx, xp):
        pupil_pixels = 128
        n_actuators = 8
        obsratio = 0.0
        diaratio = 1.0
        circGeom = False
        angleOffset = 0
        doMechCoupling = False
        couplingCoeffs = [0.31, 0.05]
        doSlaving = False
        slavingThr = 0.1
        dtype = xp.float32

        # Generate zonal influence functions
        influence_functions, pupil_mask = compute_zonal_ifunc(
            pupil_pixels,
            n_actuators,
            circ_geom=circGeom,
            angle_offset=angleOffset,
            do_mech_coupling=doMechCoupling,
            coupling_coeffs=couplingCoeffs,
            do_slaving=doSlaving,
            slaving_thr=slavingThr,
            obsratio=obsratio,
            diaratio=diaratio,
            mask=None,
            xp=xp,
            dtype=dtype,
            return_coordinates=False
        )

        # Test the dimensions of influence functions and mask
        self.assertEqual(influence_functions.shape[0], n_actuators**2)
        self.assertEqual(influence_functions.shape[1], xp.sum(pupil_mask))
        self.assertEqual(pupil_mask.shape, (pupil_pixels, pupil_pixels))
        self.assertGreater(xp.sum(pupil_mask), 0)

    @cpu_and_gpu
    def test_kl_basis_rms(self, target_device_idx, xp):
        pupil_pixels = 128
        n_actuators = 8
        telescope_diameter = 8.0
        r0 = 0.2
        L0 = 25.0
        zern_modes = 5
        oversampling = 2
        obsratio = 0.4
        diaratio = 1.0
        circGeom = True
        angleOffset = 0
        doMechCoupling = False
        couplingCoeffs = [0.31, 0.05]
        doSlaving = True
        slavingThr = 0.1
        dtype = xp.float32

        # Generate zonal influence functions
        influence_functions, pupil_mask = compute_zonal_ifunc(
            pupil_pixels,
            n_actuators,
            circ_geom=circGeom,
            angle_offset=angleOffset,
            do_mech_coupling=doMechCoupling,
            coupling_coeffs=couplingCoeffs,
            do_slaving=doSlaving,
            slaving_thr=slavingThr,
            obsratio=obsratio,
            diaratio=diaratio,
            mask=None,
            xp=xp,
            dtype=dtype,
            return_coordinates=False
        )

        # Generate the modal base
        kl_basis, _, _ = make_modal_base_from_ifs_fft(
            pupil_mask=pupil_mask,
            diameter=telescope_diameter,
            influence_functions=influence_functions,
            r0=r0,
            L0=L0,
            zern_modes=zern_modes,
            oversampling=oversampling,
            if_max_condition_number=None,
            xp=xp,
            dtype=dtype
        )

        # Test RMS of each mode
        for i, mode in enumerate(kl_basis):
            rms = xp.sqrt(xp.mean(mode**2))
            self.assertAlmostEqual(float(rms), 1.0, places=2, msg=f"Mode {i+1} RMS is not close to 1.0")
