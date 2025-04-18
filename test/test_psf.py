

import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.psf import PSF
from test.specula_testlib import cpu_and_gpu


class TestPSF(unittest.TestCase):

    @cpu_and_gpu
    def test_psf_singlewavelength(self, target_device_idx, xp):
        
        ref_S0 = 100
        t = 1
        
        psf = PSF(wavelengthInNm=500,
                  nd=3,
                  target_device_idx=target_device_idx)
        
        ef = ElectricField(120,120,0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        psf.inputs['in_ef'].set(ef)
        psf.setup(1, 1)

        psf.check_ready(t)
        psf.trigger()
        psf.post_trigger()
        sr = psf.outputs['out_sr'].value
        
        self.assertAlmostEqual(sr, 1.0)

    @cpu_and_gpu
    def test_psf_multiwavelength(self, target_device_idx, xp):
        
        ref_S0 = 100
        t = 1
        
        psf = PSF(wavelengthInNm_list=[500, 60, 700],
                  nd=2,
                  target_device_idx=target_device_idx)
        
        ef = ElectricField(120,120,0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        psf.inputs['in_ef'].set(ef)

        psf.setup(1, 1)
        psf.check_ready(t)
        psf.trigger()
        psf.post_trigger()
        sr = psf.outputs['out_sr'].value
        
        assert len(sr) == 3
        np.testing.assert_array_almost_equal(cpuArray(sr), [1.0, 1.0, 1.0])
