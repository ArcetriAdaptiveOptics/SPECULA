import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.electric_field import ElectricField
from specula.lib.make_mask import make_mask
from specula.processing_objects.extended_source import ExtendedSource
from specula.processing_objects.ext_source_pyramid import ExtSourcePyramid
from specula.processing_objects.distributed_ext_source_pyramid import DistributedExtSourcePyramid
from test.specula_testlib import cpu_and_gpu


class TestDistributedExtSourcePyramid(unittest.TestCase):

    pixel_pupil = 160
    pixel_pitch = 0.05
    wavelength_nm = 500

    def _source(self, target_device_idx, n_rings=3):
        simul_params = SimulParams(pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch)
        l_o_d = (self.wavelength_nm * 1e-9) / (self.pixel_pupil * self.pixel_pitch) * 206265
        src = ExtendedSource(simul_params=simul_params,
                             wavelengthInNm=self.wavelength_nm,
                             source_type='TOPHAT',
                             size_obj=3.0 * 4 * l_o_d,
                             sampling_type='RINGS',
                             n_rings=n_rings,
                             sampling_lambda_over_d=np.pi/6,
                             target_device_idx=target_device_idx)
        src.compute()
        return simul_params, src

    def _pyramid(self, cls, simul_params, target_device_idx, **kwargs):
        return cls(simul_params=simul_params,
                   wavelengthInNm=self.wavelength_nm,
                   fov=2.0,
                   pup_diam=30,
                   output_resolution=80,
                   max_batch_size=16,
                   target_device_idx=target_device_idx,
                   **kwargs)

    def _compare(self, target_device_idx, xp, n_slices, n_rings=3, weak_points=False,
                 n_points=None, **kwargs):
        '''
        Run an ExtSourcePyramid and a DistributedExtSourcePyramid with the same inputs
        for two steps, with different phases, and compare their outputs
        '''
        simul_params, src = self._source(target_device_idx, n_rings)
        coeff = src.outputs['coeff']
        if n_points is not None:
            # Only the first source points
            coeff = BaseValue(value=coeff.value[:n_points].copy(), target_device_idx=target_device_idx)
        if weak_points:
            # Points below the flux threshold, filtered when the streams are disabled
            coeff.value[::3, 3] *= 1e-5
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)
        ef.A = make_mask(self.pixel_pupil)

        ref = self._pyramid(ExtSourcePyramid, simul_params, target_device_idx, **kwargs)
        dist = self._pyramid(DistributedExtSourcePyramid, simul_params, target_device_idx,
                             n_slices=n_slices, **kwargs)
        for pyr in (ref, dist):
            pyr.inputs['in_ef'].set(ef)
            pyr.inputs['ext_source_coeff'].set(coeff)

        rng = np.random.default_rng(1)
        for t in (1, 2):
            ef.phaseInNm[:] = xp.asarray(100 * rng.standard_normal((self.pixel_pupil, self.pixel_pupil)))
            ef.generation_time = t
            coeff.generation_time = t
            for pyr in (ref, dist):
                if t == 1:
                    pyr.setup()
                pyr.check_ready(t)
                pyr.trigger()
                pyr.post_trigger()

            for name in ('out_i', 'out_psf_tot', 'out_psf_bfm', 'out_transmission'):
                out_ref = ref.outputs[name]
                out_dist = dist.outputs[name]
                a = out_ref.i if name == 'out_i' else out_ref.value
                b = out_dist.i if name == 'out_i' else out_dist.value
                np.testing.assert_allclose(cpuArray(b), cpuArray(a), rtol=1e-4,
                                           atol=1e-6 * float(np.abs(cpuArray(a)).max()),
                                           err_msg=f'{name} at step {t}')
                self.assertEqual(out_dist.generation_time, t)
        return dist

    @cpu_and_gpu
    def test_same_result_as_ext_source_pyramid(self, target_device_idx, xp):
        dist = self._compare(target_device_idx, xp, n_slices=3)
        # All points are computed once, in some slice
        n_points = dist.sub_pyr[0].mod_steps
        self.assertEqual(sum(len(cpuArray(p.valid_idx)) for p in dist.sub_pyr), n_points)
        self.assertGreater(n_points, 3 * 16)    # more than one chunk per slice

    @cpu_and_gpu
    def test_same_result_with_flux_filtering(self, target_device_idx, xp):
        self._compare(target_device_idx, xp, n_slices=2, weak_points=True,
                      cuda_stream_enable=False)

    @cpu_and_gpu
    def test_empty_slices(self, target_device_idx, xp):
        # More slices than source points: the first slice computes nothing
        dist = self._compare(target_device_idx, xp, n_slices=4, n_points=3)
        self.assertEqual([len(cpuArray(p.valid_idx)) for p in dist.sub_pyr], [0, 1, 1, 1])

    def test_default_device(self):
        # Without target_device_idx, the object and its first slice are on the default device
        simul_params, _ = self._source(None, n_rings=0)
        dist = self._pyramid(DistributedExtSourcePyramid, simul_params, None, n_slices=2)
        self.assertEqual(dist.sub_pyr[0].target_device_idx, dist.target_device_idx)

    def test_allocator_hooks(self):
        params = {'class': 'DistributedExtSourcePyramid', 'n_slices': 4, 'target_device_idx': 0}
        self.assertEqual(DistributedExtSourcePyramid.devices_used(params), 4)
        self.assertEqual(DistributedExtSourcePyramid.single_device_params(params)['n_slices'], 1)
        self.assertEqual(params['n_slices'], 4)

    def test_invalid_slice(self):
        simul_params, _ = self._source(-1, n_rings=0)
        with self.assertRaises(ValueError):
            self._pyramid(ExtSourcePyramid, simul_params, -1, n_point_slices=2, point_slice_idx=2)

    def test_same_defaults_as_ext_source_pyramid(self):
        import inspect
        base = inspect.signature(ExtSourcePyramid.__init__).parameters
        for name, p in inspect.signature(DistributedExtSourcePyramid.__init__).parameters.items():
            if name == 'n_slices':
                continue
            self.assertIn(name, base)
            self.assertEqual(p.default, base[name].default, name)

    def test_split_options(self):
        params = {'class': 'ExtSourcePyramid', 'simul_params_ref': 'main', 'wavelengthInNm': 750,
                  'fov': 2.0, 'pup_diam': 30, 'output_resolution': 80, 'max_batch_size': 100,
                  'inputs': {'in_ef': 'prop.out_ef', 'ext_source_coeff': 'src.coeff'}}
        options = ExtSourcePyramid.split_options(params, 4)
        self.assertEqual(sorted(options), [2, 3, 4])
        for k, split in options.items():
            self.assertEqual(split['class'], 'DistributedExtSourcePyramid')
            self.assertEqual(split['n_slices'], k)
            self.assertEqual(DistributedExtSourcePyramid.devices_used(split), k)
            self.assertEqual(split['inputs'], params['inputs'])
        # Smaller batches, so that each slice computes a k-th of the padded points
        self.assertEqual([options[k]['max_batch_size'] for k in (2, 3, 4)], [50, 34, 25])
        self.assertEqual(params['class'], 'ExtSourcePyramid')     # not modified
        self.assertEqual(params['max_batch_size'], 100)
        # Default batch size
        no_batch = {k: v for k, v in params.items() if k != 'max_batch_size'}
        self.assertEqual(ExtSourcePyramid.split_options(no_batch, 2)[2]['max_batch_size'], 64)
        self.assertEqual(ExtSourcePyramid.split_options(params, 1), {})
        # Not split: parameters that DistributedExtSourcePyramid does not accept, a slice
        # of points already given, and DistributedExtSourcePyramid itself
        self.assertEqual(ExtSourcePyramid.split_options(dict(params, unknown=1), 4), {})
        self.assertEqual(ExtSourcePyramid.split_options(dict(params, n_point_slices=2), 4), {})
        self.assertEqual(DistributedExtSourcePyramid.split_options(dict(params, n_slices=2), 4), {})
