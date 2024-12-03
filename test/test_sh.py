

import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.data_objects.ef import ElectricField
from specula.processing_objects.sh import SH


class TestSH(unittest.TestCase):

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_sh_flux_gpu(self):
        self._test_sh_flux(target_device_idx=0, xp=cp)

    def test_sh_flux_cpu(self):
        self._test_sh_flux(target_device_idx=-1, xp=np)
        
    def _test_sh_flux(self, target_device_idx, xp):
        
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

        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        intensity = sh.outputs['out_i']
        
        np.testing.assert_almost_equal(xp.sum(intensity.i), ref_S0 * ef.masked_area())

