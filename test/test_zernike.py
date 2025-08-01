import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula.lib.zernike_generator import ZernikeGenerator
from test.specula_testlib import cpu_and_gpu
from specula import cpuArray

class TestZernikeGenerator(unittest.TestCase):
    def setUp(self):
        self.size = 64
        self.plot_debug = True  # Set to True to enable plotting for debugging

    @cpu_and_gpu
    def test_tip_and_tilt_shape(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        tip = zg.getZernike(2)
        tilt = zg.getZernike(3)
        coma = zg.getZernike(7)
        if self.plot_debug:
            import matplotlib.pyplot as plt
            # Extract data properly for plotting
            if hasattr(tip, 'data'):
                tip_plot = cpuArray(tip.data)
                tilt_plot = cpuArray(tilt.data)
                coma_plot = cpuArray(coma.data)
            else:
                tip_plot = cpuArray(tip)
                tilt_plot = cpuArray(tilt)
                coma_plot = cpuArray(coma)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(tip_plot, cmap='gray')
            plt.title('Tip')
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(tilt_plot, cmap='gray')
            plt.title('Tilt')
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(coma_plot, cmap='gray')
            plt.title('Coma')
            plt.colorbar()
            plt.show()

        self.assertEqual(tip.shape, (self.size, self.size))
        self.assertEqual(tilt.shape, (self.size, self.size))
        self.assertEqual(coma.shape, (self.size, self.size))

    @cpu_and_gpu
    def test_masked_area(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        tip = zg.getZernike(2)
        mask = zg._boolean_mask
        
        # Handle both masked arrays (CPU) and regular arrays (GPU)
        if hasattr(tip, 'data'):
            tip_np = cpuArray(tip.data)
            mask_np = cpuArray(tip.mask)  # Use the mask from the masked array
        else:
            tip_np = cpuArray(tip)
            mask_np = cpuArray(mask)
        
        # Outside the disk, values should be zero
        self.assertTrue(np.all(tip_np[mask_np] == 0))

    @cpu_and_gpu
    def test_piston_constant(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        piston = zg.getZernike(1)
        mask = zg._boolean_mask
        piston_np = cpuArray(piston.data if hasattr(piston, 'data') else piston)
        mask_np = cpuArray(mask)  # Ensure mask is also numpy
        in_disk = ~mask_np
        # The value should be constant inside the disk
        self.assertAlmostEqual(float(np.std(piston_np[in_disk])), 0, places=10)

    @cpu_and_gpu
    def test_norm(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        mask = zg._boolean_mask
        mask_np = cpuArray(mask)  # Ensure mask is also numpy
        for idx in range(1, 5):
            z = zg.getZernike(idx)
            z_np = cpuArray(z.data if hasattr(z, 'data') else z)
            in_disk = ~mask_np
            norm = float(np.sqrt(np.sum(z_np[in_disk]**2) / np.sum(in_disk)))
            self.assertAlmostEqual(norm, 1, places=2)