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
        self.plot_debug = False  # Set to True to enable plotting for debugging

    @cpu_and_gpu
    def test_tip_and_tilt_shape(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        tip = zg.getZernike(2)
        tilt = zg.getZernike(3)
        coma = zg.getZernike(7)
        
        if self.plot_debug:
            import matplotlib.pyplot as plt
            # Extract data properly for plotting - always convert to numpy
            tip_plot = cpuArray(tip.data if hasattr(tip, 'data') else tip)
            tilt_plot = cpuArray(tilt.data if hasattr(tilt, 'data') else tilt)
            coma_plot = cpuArray(coma.data if hasattr(coma, 'data') else coma)

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

        # Handle both masked arrays (CPU) and regular arrays (GPU)
        if hasattr(tip, 'data') and hasattr(tip, 'mask'):
            # CPU: masked array
            tip_np = cpuArray(tip.data)
            mask_np = cpuArray(tip.mask)
        else:
            # GPU: regular array
            tip_np = cpuArray(tip)
            mask_np = cpuArray(zg._boolean_mask)

        # Inside the disk (where mask_np is False), values should be non-zero for tip/tilt
        in_disk = ~mask_np
        # Check that we have some non-zero values inside the disk
        self.assertTrue(np.any(np.abs(tip_np[in_disk]) > 1e-6))
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(tip_np[in_disk])))
        
        # Check that we have the right shape
        self.assertEqual(tip_np.shape, (self.size, self.size))

    @cpu_and_gpu
    def test_piston_constant(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        piston = zg.getZernike(1)

        # Handle both masked arrays (CPU) and regular arrays (GPU)
        if hasattr(piston, 'data') and hasattr(piston, 'mask'):
            # CPU: masked array
            piston_np = cpuArray(piston.data)
            mask_np = cpuArray(piston.mask)
        else:
            # GPU: regular array
            piston_np = cpuArray(piston)
            mask_np = cpuArray(zg._boolean_mask)

        in_disk = ~mask_np
        # The value should be constant inside the disk
        self.assertAlmostEqual(float(np.std(piston_np[in_disk])), 0, places=10)

    @cpu_and_gpu
    def test_norm(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)

        # Get the mask once
        mask_np = cpuArray(zg._boolean_mask)
        in_disk = ~mask_np

        for idx in range(1, 5):
            z = zg.getZernike(idx)

            # Handle both masked arrays (CPU) and regular arrays (GPU)
            if hasattr(z, 'data') and hasattr(z, 'mask'):
                # CPU: masked array
                z_np = cpuArray(z.data)
            else:
                # GPU: regular array
                z_np = cpuArray(z)

            norm = float(np.sqrt(np.sum(z_np[in_disk]**2) / np.sum(in_disk)))
            self.assertAlmostEqual(norm, 1, places=2)