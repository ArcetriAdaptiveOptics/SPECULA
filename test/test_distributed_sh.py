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
        
    def test_default_device(self):
        # Without target_device_idx, the object and its first slice are on the default device
        sh = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov=3,
                sensor_pxscale=0.5,
                subap_on_diameter=20,
                subap_npx=6,
                n_slices=2)
        self.assertEqual(sh.sub_sh[0].target_device_idx, sh.target_device_idx)

    def test_allocator_hooks(self):
        params = {'class': 'DistributedSH', 'n_slices': 4, 'target_device_idx': 0}
        self.assertEqual(DistributedSH.devices_used(params), 4)
        self.assertEqual(DistributedSH.single_device_params(params)['n_slices'], 1)
        self.assertEqual(params['n_slices'], 4)

    def test_repeated_work_fraction_decreases_with_the_rows(self):
        def f(rows):
            return DistributedSH.repeated_work_fraction({'subap_on_diameter': rows})
        self.assertEqual(f(2), 0.22)
        self.assertEqual(f(8), 0.22)
        self.assertEqual(f(68), 0.05)
        self.assertEqual(f(80), 0.05)
        self.assertTrue(0.22 > f(10) > f(40) > 0.05)

    def test_slices_prepare_after_fetching_their_inputs(self):
        '''
        At each step each slice prepares its trigger once, after getting its
        inputs. SH initializes the extrapolation of the input field at its first
        prepare_trigger(): a slice on another device than its input, which reads a
        copy of it, would initialize it from the copy of the previous step (at the
        first step, the field before its producer's first trigger, e.g. a field
        received by MPI, sent before the producer's setup).
        '''
        from specula.data_objects.electric_field import ElectricField
        n = 120
        ef = ElectricField(n, n, 0.0802 * 4, S0=100.0, target_device_idx=-1)
        ef.generation_time = 0
        sh = DistributedSH(wavelengthInNm=750, subap_wanted_fov=4.6, sensor_pxscale=0.153,
                           subap_on_diameter=8, subap_npx=30, n_slices=4, target_device_idx=-1)
        sh.inputs['in_ef'].set(ef)
        sh.setup()
        calls = {i: [] for i in range(len(sh.sub_sh))}
        for i, sub in enumerate(sh.sub_sh):
            def record(method, event, i=i):
                def wrapper(*args, **kwargs):
                    calls[i].append(event)
                    return method(*args, **kwargs)
                return wrapper
            sub.get_all_inputs = record(sub.get_all_inputs, 'inputs')
            sub.prepare_trigger = record(sub.prepare_trigger, 'prepare')
        sh.check_ready(0)
        for i in calls:
            self.assertEqual(calls[i], ['inputs', 'prepare'], f'slice {i}')

    def test_sh_split_options(self):
        from specula.processing_objects.sh import SH
        params = {'class': 'SH', 'wavelengthInNm': 589, 'subap_wanted_fov': 16.1,
                  'sensor_pxscale': 1.15, 'subap_on_diameter': 68, 'subap_npx': 14,
                  'fov_ovs_coeff': 1.2, 'laser_launch_tel_ref': 'llt',
                  'inputs': {'in_ef': 'prop.out_ef'}}
        options = SH.split_options(params, 4)
        # The divisors of the 68 subaperture rows, up to the GPUs of a node
        self.assertEqual(sorted(options), [2, 4])
        for k, split in options.items():
            self.assertEqual(split['class'], 'DistributedSH')
            self.assertEqual(split['n_slices'], k)
            self.assertEqual(DistributedSH.devices_used(split), k)
            self.assertEqual(split['fov_ovs_coeff'], 1.2)
            self.assertEqual(split['inputs'], params['inputs'])
        self.assertEqual(params['class'], 'SH')         # not modified
        self.assertEqual(sorted(SH.split_options(params, 8)), [2, 4])
        self.assertEqual(SH.split_options(dict(params, subap_on_diameter=7), 4), {})
        self.assertEqual(SH.split_options(params, 1), {})
        # DistributedSH has the defaults of SH, so none is added to the parameters
        import inspect
        sh_defaults = inspect.signature(SH.__init__).parameters
        for name, p in inspect.signature(DistributedSH.__init__).parameters.items():
            if name in sh_defaults and p.default is not inspect.Parameter.empty:
                self.assertEqual(p.default, sh_defaults[name].default, name)
        no_fov = {k: v for k, v in params.items() if k != 'fov_ovs_coeff'}
        self.assertNotIn('fov_ovs_coeff', SH.split_options(no_fov, 2)[2])
        # Not split: parameters that DistributedSH does not accept, a slice of rows
        # already given, and DistributedSH itself (its number of slices is kept)
        self.assertEqual(SH.split_options(dict(params, data_dir='x'), 4), {})
        self.assertEqual(SH.split_options(dict(params, subap_rows_slice=[0, 10]), 4), {})
        self.assertEqual(DistributedSH.split_options(dict(params, n_slices=2), 4), {})

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
    def test_uneven_slices(self, target_device_idx, xp):
        '''
        Test that all subaperture rows are computed when the number
        of subapertures is not a multiple of the number of slices,
        comparing with a single slice
        '''
        t = 1
        params = dict(wavelengthInNm=500,
                      subap_wanted_fov=3,
                      sensor_pxscale=0.5,
                      subap_on_diameter=20,
                      subap_npx=6,
                      target_device_idx=target_device_idx)
        dsh = DistributedSH(n_slices=3, **params)
        ref = DistributedSH(n_slices=1, **params)

        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        ef.generation_time = t
        for obj in [dsh, ref]:
            obj.inputs['in_ef'].set(ef)
            obj.setup()
            obj.check_ready(t)
            obj.trigger()
            obj.post_trigger()

        self.assertEqual(dsh.slices[-1].stop, 20)
        np.testing.assert_allclose(cpuArray(dsh.outputs['out_i'].i),
                                   cpuArray(ref.outputs['out_i'].i), rtol=1e-5, atol=1e-6)

    @cpu_and_gpu
    def test_pixelscale(self, target_device_idx, xp):
        '''
        Test that pixelscale is correctly handled, by comparing spots from a flat 
        wavefront and from a tilted one. The introduced tilt corresponds to exactly 1 pixel,
        and we verify that the resulting intensity field is indeed shifted by 1 pixel
        in the correct direction.

        The pupil sampling and fov_ovs_coeff=1 are chosen so that the input field is not
        interpolated: the interpolation magnifies the field by N/(N-1) (issue #838),
        which would shift the spot by (N-1)/N pixels only.
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 240
        pixel_pitch = 0.025
        sh_npix = 6

        sh = DistributedSH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                n_slices=2,
                fov_ovs_coeff=1.0,
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
