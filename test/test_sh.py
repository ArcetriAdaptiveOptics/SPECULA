import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH
from test.specula_testlib import cpu_and_gpu


class TestSH(unittest.TestCase):

    @cpu_and_gpu
    def test_sh_flux(self, target_device_idx, xp):

        ref_S0 = 100
        t = 1

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov=3,
                sensor_pxscale=0.5,
                subap_on_diameter=20,
                subap_npx=6,
                target_device_idx=target_device_idx)

        ef = ElectricField(120,120,0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        sh.inputs['in_ef'].set(ef)

        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        intensity = sh.outputs['out_i']

        np.testing.assert_almost_equal(xp.sum(intensity.i), ref_S0 * ef.masked_area())

    @cpu_and_gpu
    def test_pixelscale(self, target_device_idx, xp):
        '''
        Test that pixelscale is correctly handled, by comparing spots from a flat 
        wavefront and from a tilted one. The introduced tilt corresponds to exactly 1 pixel,
        and we verify that the resulting intensity field is indeed shifted by 1 pixel
        in the correct direction
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        flat = sh.outputs['out_i'].i.copy()

        # tilt corresponding to pxscale_arcsec
        tilt_value = np.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = np.linspace(-tilt_value / 2 * (1-1/pixel_pupil), tilt_value / 2 * (1-1/pixel_pupil), pixel_pupil)

        # Tilted wavefront
        ef.phaseInNm[:] = xp.array(np.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        ef.generation_time = t+1

        sh.check_ready(t+1)
        sh.trigger()
        sh.post_trigger()
        tilted = sh.outputs['out_i'].i.copy()

        flat_shifted = np.roll(flat, (0, 1))

        # Remove the left column edges on each subap (comparison is invalid after roll)
        flat_shifted[:, ::sh_npix] = 0
        tilted[:, ::sh_npix] = 0

        # import matplotlib.pyplot as plt
        # plt.imshow(cpuArray(tilted))
        # plt.figure()
        # plt.imshow(cpuArray(flat_shifted))
        # plt.show()

        np.testing.assert_array_almost_equal(cpuArray(tilted), cpuArray(flat_shifted), decimal=4)

    @cpu_and_gpu
    def test_zeros_cache(self, target_device_idx, xp):
        '''
        Test that arrays are re-used between SH instances on the same target
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6

        # clear cache before test
        SH._SH__zeros_cache.clear()

        sh1 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        sh2 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        sh3 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=30,  # Different
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)


        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil,
                           pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)
        ef.generation_time = t
        sh1.inputs['in_ef'].set(ef)
        sh2.inputs['in_ef'].set(ef)
        sh3.inputs['in_ef'].set(ef)

        sh1.setup()
        sh2.setup()
        sh3.setup()

        # Test 1: sh1 and sh2 should share arrays (same geometry, same rank)
        assert id(sh1._wf3) == id(sh2._wf3), "sh1 and sh2 should share _wf3"
        assert id(sh1.psf) == id(sh2.psf), "sh1 and sh2 should share psf"
        assert id(sh1.psf_shifted) == id(sh2.psf_shifted), "sh1 and sh2 should share psf_shifted"
        assert id(sh1.ef_row) == id(sh2.ef_row), "sh1 and sh2 should share ef_row"

        # Test 2: sh3 should NOT share with sh1/sh2 (different geometry)
        assert id(sh1._wf3) != id(sh3._wf3), "sh3 should have different _wf3 (different geometry)"

        # Test 4: Check cache size
        cache_size = len(SH._SH__zeros_cache)
        self.assertGreater(cache_size, 0, "Cache should have entries")

        # We should have entries for:
        # - sh1/sh2 (shared, rank 0, geometry 20)
        # - sh3 (separate, rank 0, geometry 30)
        # Each geometry allocates 4 arrays
        #  (_wf3, psf==psf_shifted, ef_row, _psfimage, _psf_reshaped_2d)
        # So expected: 2 geometries × 5 arrays = 10 entries
        assert cache_size == 10
        print(f"Cache has {cache_size} entries")

    @cpu_and_gpu
    def test_oversampling_alignment(self, target_device_idx, xp):
        '''
        Test that the new float oversampling logic correctly aligns the phase size
        to be a multiple of (2 * n_lenses), even if the input size is irregular.
        '''
        t = 1
        wl = 500 # nm
        # We simulate a case where the pupil is 105 pixels and we want 10 subaps.
        # Modulus required = 2 * 10 = 20.
        # 105 is NOT divisible by 20. Next multiple is 120.
        # Expected oversampling = 120 / 105 = 1.142857...

        pixel_pupil = 105 # Irregular size
        n_lenses = 10

        sh = SH(wavelengthInNm=wl,
                subap_wanted_fov=2.0,
                sensor_pxscale=0.5,
                subap_on_diameter=n_lenses,
                subap_npx=4,
                fov_ovs_coeff=1.0, # No forced coeff, we want to test the automatic adjustment
                target_device_idx=target_device_idx)

        # Create the irregular electric field
        ef = ElectricField(pixel_pupil, pixel_pupil, 0.05, S0=1,
                           target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()

        # 1. Check if the oversampling factor is a float > 1.0
        self.assertGreater(sh._fov_ovs, 1.0, "Oversampling should be > 1.0 to fix alignment")

        # 2. Verify the math: 105 * ovs should be exactly 120
        calculated_size = pixel_pupil * sh._fov_ovs
        self.assertAlmostEqual(calculated_size, 120.0, places=5,
                               msg=f"Expected 120 total pixels, got {calculated_size}")

        # 3. Verify internal pixel count
        # With Lenslet diameter normalization = 2.0 (standard implied by 12!=6):
        # lens[2] = 2/n_lenses = 0.2
        # _ovs_np_sub = round(120 * 0.2 * 0.5) = round(12.0) = 12
        # This represents the full subaperture width in pixels (120 pixels / 10 subaps).
        self.assertEqual(sh._ovs_np_sub, 12,
                         "Internal subap pixel count should match total/n_lenses")

    @cpu_and_gpu
    def test_oversampling_forced_coeff(self, target_device_idx, xp):
        '''
        Test that providing a specific fov_ovs_coeff works and still respects
        the geometry constraints (multiple of 2*n_lenses).
        '''
        t = 1
        pixel_pupil = 100
        n_lenses = 10
        # Modulus = 20.

        # We force coefficient = 1.5
        # Target minimum size = 100 * 1.5 = 150.
        # 150 is NOT divisible by 20 (150/20 = 7.5).
        # Next multiple of 20 is 160.
        # Expected final oversampling = 160 / 100 = 1.6

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov=2.0,
                sensor_pxscale=0.5,
                subap_on_diameter=n_lenses,
                subap_npx=4,
                fov_ovs_coeff=1.5, # FORCE THIS
                target_device_idx=target_device_idx)

        ef = ElectricField(pixel_pupil, pixel_pupil, 0.05, S0=1,
                           target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()

        # Check that we respected the forced coeff (at least)
        self.assertGreaterEqual(sh._fov_ovs, 1.5, "Should respect minimum forced coefficient")

        # Check that we adjusted for geometry (1.6 expected)
        self.assertAlmostEqual(sh._fov_ovs, 1.6, places=5,
                               msg="Should have adjusted 1.5 -> 1.6 for geometry alignment")

        # Verify final size
        final_size = pixel_pupil * sh._fov_ovs
        self.assertAlmostEqual(final_size % 20, 0, places=5,
                               msg="Final size must be divisible by 20")
