import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula import cp, cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.processing_objects.poly_chrom_sh import PolyChromSH
from specula.processing_objects.poly_crom_pyramid import PolyCromPyramid
from test.specula_testlib import cpu_and_gpu


class TestPolyChromDevices(unittest.TestCase):
    '''
    The polychromatic WFS with the instances of the wavelengths spread
    over several devices (n_devices) give the same result as on one device
    '''
    wavelengths = [500, 600, 700]
    flux_factors = [0.8, 1.0, 0.6]
    xy_tilts = [[0, 0], [0.05, 0.02], [-0.03, 0.08]]

    def _sh(self, target_device_idx, n_devices):
        return PolyChromSH(wavelengthInNm=self.wavelengths,
                           flux_factor=self.flux_factors,
                           xy_tilts_in_arcsec=self.xy_tilts,
                           subap_wanted_fov=2,
                           sensor_pxscale=0.1,
                           subap_on_diameter=4,
                           subap_npx=20,
                           n_devices=n_devices,
                           target_device_idx=target_device_idx)

    def _pyr(self, target_device_idx, n_devices):
        return PolyCromPyramid(wavelengthInNm=self.wavelengths,
                               flux_factor=self.flux_factors,
                               xy_tilts_in_arcsec=self.xy_tilts,
                               simul_params=SimulParams(pixel_pupil=120, pixel_pitch=0.05),
                               fov=2.0,
                               pup_diam=30,
                               output_resolution=80,
                               mod_amp=3.0,
                               n_devices=n_devices,
                               target_device_idx=target_device_idx)

    def _compare(self, make, target_device_idx, xp):
        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(120)
        ref = make(target_device_idx, n_devices=1)
        dist = make(target_device_idx, n_devices=2)
        for wfs in (ref, dist):
            wfs.inputs['in_ef'].set(ef)

        if xp is cp and cp.cuda.runtime.getDeviceCount() > 1:
            # Wavelength i on device target_device_idx + i % 2
            self.assertEqual([w.target_device_idx for w in dist._wfs_instances], [0, 1, 0])
        self.assertTrue(all(w.target_device_idx == ref.target_device_idx
                            for w in ref._wfs_instances))

        rng = np.random.default_rng(1)
        for t in (1, 2):
            ef.phaseInNm[:] = xp.asarray(50 * rng.standard_normal((120, 120)))
            ef.generation_time = t
            for wfs in (ref, dist):
                if t == 1:
                    wfs.setup()
                wfs.check_ready(t)
                wfs.trigger()
                wfs.post_trigger()
            np.testing.assert_allclose(cpuArray(dist.outputs['out_i'].i),
                                       cpuArray(ref.outputs['out_i'].i),
                                       rtol=1e-5, atol=1e-6 * float(ref.outputs['out_i'].i.max()),
                                       err_msg=f'step {t}')
            for i in range(len(self.wavelengths)):
                np.testing.assert_allclose(cpuArray(dist.get_wavelength_contribution(i)),
                                           cpuArray(ref.get_wavelength_contribution(i)),
                                           rtol=1e-5, atol=1e-6 * float(ref.outputs['out_i'].i.max()))

    @cpu_and_gpu
    def test_poly_chrom_sh_devices(self, target_device_idx, xp):
        self._compare(self._sh, target_device_idx, xp)

    @cpu_and_gpu
    def test_poly_crom_pyramid_devices(self, target_device_idx, xp):
        self._compare(self._pyr, target_device_idx, xp)

    def test_allocator_hooks(self):
        for cls in (PolyChromSH, PolyCromPyramid):
            params = {'class': cls.__name__, 'n_devices': 3, 'target_device_idx': 0}
            self.assertEqual(cls.devices_used(params), 3)
            self.assertEqual(cls.single_device_params(params)['n_devices'], 1)
            self.assertEqual(params['n_devices'], 3)
            self.assertEqual(cls.devices_used({'class': cls.__name__}), 1)

    def test_split_options(self):
        for cls in (PolyChromSH, PolyCromPyramid):
            params = {'class': cls.__name__, 'wavelengthInNm': [500, 600, 700],
                      'flux_factor': [1, 1, 1], 'inputs': {'in_ef': 'prop.out_ef'}}
            options = cls.split_options(params, 4)
            # Not more devices than wavelengths
            self.assertEqual(sorted(options), [2, 3])
            for k, split in options.items():
                self.assertEqual(split['n_devices'], k)
                self.assertEqual(split['class'], cls.__name__)
                self.assertEqual(cls.devices_used(split), k)
                self.assertEqual(split['inputs'], params['inputs'])
            self.assertNotIn('n_devices', params)     # not modified
            self.assertEqual(cls.split_options(params, 1), {})
            # An explicit n_devices is kept
            self.assertEqual(cls.split_options(dict(params, n_devices=1), 4), {})

    def test_repeated_work_fraction(self):
        def part(n, k):
            # Time of each part, as a fraction of the whole object, as in the allocator
            f = PolyChromSH.repeated_work_fraction({'wavelengthInNm': list(range(n)), 'n_devices': k})
            return f + (1 - f) / k
        # The busiest device computes ceil(n/k) of the n wavelengths
        self.assertAlmostEqual(part(3, 2), 2 / 3)
        self.assertAlmostEqual(part(4, 2), 1 / 2)
        self.assertAlmostEqual(part(3, 3), 1 / 3)
        self.assertAlmostEqual(part(5, 3), 2 / 5)
        self.assertEqual(PolyChromSH.repeated_work_fraction({'wavelengthInNm': [1, 2]}), 0.0)

    def test_invalid_n_devices(self):
        with self.assertRaises(ValueError):
            self._sh(-1, n_devices=0)
