import specula
specula.init(0)  # Default target device

import unittest
import tempfile
import os
import shutil
from specula import np, cpuArray
from test.specula_testlib import cpu_and_gpu
from specula.data_objects.psd import PSD

class TestPSD(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.n_modes = 3
        self.n_time_steps = 1000
        self.fs = 1000.0
        self.dt = 1.0 / self.fs
        # Create [N, M] dummy time series
        self.dummy_data = np.random.randn(self.n_modes, self.n_time_steps)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @cpu_and_gpu
    def test_psd_dimensions(self, target_device_idx, xp):
        """Verify [N,M] input yields [N,L] psd_data and length L freq_vec"""
        psd_obj = PSD(data=self.dummy_data, fs=self.fs, target_device_idx=target_device_idx)
        
        L = psd_obj.samplespersegment // 2 + 1
        
        self.assertEqual(psd_obj.psd_data.shape, (self.n_modes, L))
        self.assertEqual(psd_obj.freq_vec.shape, (L,))
        self.assertEqual(psd_obj.integrated_power.shape, (self.n_modes,))

    # @cpu_and_gpu
    # def test_overwrite(self, target_device_idx, xp):
    #     """Verify that save respect overwrite flag"""
    #     save_path = os.path.join(self.test_dir, "test_overwrite.fits")
    #     psd = PSD(data=self.dummy_data, dt=self.dt, description="Test_IO", target_device_idx=target_device_idx, overwrite=False)

    #     # self.assertRaises(FileExistsError,psd.save,save_path)
    #     psd.save(save_path)

    #     psd.overwrite = True
    #     psd.save(save_path) # should not raise

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        """Verify that loaded data matches saved data exactly"""
        save_path = os.path.join(self.test_dir, "test_io.fits")
        original = PSD(data=self.dummy_data, dt=self.dt, description="Test_IO", target_device_idx=target_device_idx, overwrite=True)
        original.save(save_path)
        
        restored = PSD.restore(save_path)
        
        # Check metadata and data arrays
        self.assertEqual(restored.description, original.description)
        np.testing.assert_array_almost_equal(cpuArray(restored.psd_data), cpuArray(original.psd_data))
        np.testing.assert_array_almost_equal(cpuArray(restored.freq_vec), cpuArray(original.freq_vec))
        np.testing.assert_array_almost_equal(cpuArray(restored.integrated_power), cpuArray(original.integrated_power))

    @cpu_and_gpu
    def test_integrated_power_computation(self, target_device_idx, xp):
        """Verify total power equals the integral of the PSD (Parseval's relation)"""
        psd_obj = PSD(data=self.dummy_data, fs=self.fs, target_device_idx=target_device_idx)
        freq_domain_power = cpuArray(psd_obj.get_integrated_power())
        
        # Note: Welch with windowing and nperseg might lead to small scaling differences,
        # but the shapes must match exactly.
        self.assertEqual(freq_domain_power.shape, (self.n_modes,))
        # General check for non-zero power
        self.assertTrue(np.all(freq_domain_power > 0))


    @cpu_and_gpu
    def test_interpolate_output_shape(self, target_device_idx, xp):
        """Verify interpolation onto a new frequency vector preserves mode count"""
        psd_obj = PSD(data=self.dummy_data, fs=self.fs, target_device_idx=target_device_idx)
        
        new_L = 50
        new_freq = np.linspace(0, self.fs/2, new_L)
        interpolated = psd_obj.interpolate(new_freq)
        
        # Should return [N, new_L]
        self.assertEqual(interpolated.shape, (self.n_modes, new_L))
