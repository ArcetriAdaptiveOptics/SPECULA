

import specula
specula.init(0)  # Default target device

import unittest

from specula.lib.zernike_generator import ZernikeGenerator
from test.specula_testlib import cpu_and_gpu

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
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(tip.data, cmap='gray')
            plt.title('Tip')
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(tilt.data, cmap='gray')
            plt.title('Tilt')
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(coma.data, cmap='gray')
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
        # The mask is True outside the disk
        y, x = xp.ogrid[:self.size, :self.size]
        mask = ((y - self.size/2 + 0.5)**2 + (x - self.size/2 + 0.5)**2) > (self.size/2)**2
        # The mask of tip should be True outside the disk
        self.assertTrue(xp.all(tip.mask[mask]))

    @cpu_and_gpu
    def test_piston_constant(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        piston = zg.getZernike(1)
        # The value should be constant inside the disk
        in_disk = ~piston.mask
        self.assertAlmostEqual(float(xp.std(piston.data[in_disk])), 0, places=10)

    @cpu_and_gpu
    def test_norm(self, target_device_idx, xp):
        zg = ZernikeGenerator(self.size, xp=xp, dtype=xp.float32)
        # The L2 norm of Zernike polynomials (inside the disk) should be ~1
        for idx in range(1, 5):
            z = zg.getZernike(idx)
            in_disk = ~z.mask
            norm = float(xp.sqrt(xp.sum(z.data[in_disk]**2) / xp.sum(in_disk)))
            self.assertAlmostEqual(norm, 1, places=2)