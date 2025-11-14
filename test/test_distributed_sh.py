

import specula
specula.init(0)  # Default target device

import os
import sys
import unittest
import tempfile
import subprocess

from specula import cp, np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.distributed_sh import DistributedSH
from test.specula_testlib import cpu_and_gpu


class TestDistributedSH(unittest.TestCase):
    '''
    Same tests as TestSH, but with the DistributedSH class
    '''
    @cpu_and_gpu
    def test_distributed_sh_slices_with_one_GPU(self, target_device_idx, xp):
        '''
        Test that sub-SH objects are allocated on the same GPU
        if multiple slices are requested but only one GPU is present
        '''
        # Only run on GPU
        if xp != cp:
            return

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write('''
import specula
specula.init(0)  # Default target device
from specula.processing_objects.distributed_sh import DistributedSH
sh = DistributedSH(wavelengthInNm=500,
        subap_wanted_fov=3,
        sensor_pxscale=0.5,
        subap_on_diameter=20,
        subap_npx=6,
        n_slices=2,
        target_device_idx=0)
print(f'{sh.sub_sh[0].target_device_idx},{sh.sub_sh[1].target_device_idx}')
''')
            script_path = f.name

        # Create isolated environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertEqual(result.stdout.splitlines()[-1].strip(), "0,0")
        
    @cpu_and_gpu
    def test_distributed_sh_flux(self, target_device_idx, xp):
        
        ref_S0 = 100
        t = 1
        sh = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov=3,
                sensor_pxscale=0.5,
                subap_on_diameter=20,
                subap_npx=6,
                n_slices=2,
                target_device_idx=target_device_idx)
        ef = ElectricField(120,120,0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        intensity = sh.outputs['out_i']
        
        if xp == cp:
            cp.cuda.Device(target_device_idx).use()
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

        sh = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                n_slices=2,
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

        sh1 = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                n_slices=2,
                target_device_idx=target_device_idx)

        sh2 = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                n_slices=2,
                target_device_idx=target_device_idx)

        sh3 = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=30,  # Different
                subap_npx=sh_npix,
                n_slices=2,
                target_device_idx=target_device_idx)

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh1.inputs['in_ef'].set(ef)
        sh2.inputs['in_ef'].set(ef)
        sh3.inputs['in_ef'].set(ef)

        sh1.setup()
        sh2.setup()
        sh3.setup()
        
        assert id(sh1.sub_sh[0]._wf3) == id(sh2.sub_sh[0]._wf3) 
        assert id(sh1.sub_sh[1]._wf3) != id(sh3.sub_sh[1]._wf3) 
