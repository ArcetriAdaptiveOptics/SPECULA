import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray

from iteras.lib.rebin import rebin2d
from test.specula_testlib import cpu_and_gpu


class TestRebin2d(unittest.TestCase):

    @cpu_and_gpu
    def test_downsampling_is_block_mean(self, target_device_idx, xp):
        '''Same as the mean over the blocks, also for non-square shapes,
        different factors on the two axes and integer inputs'''
        for dtype in (np.float32, np.float64, np.int32):
            for in_shape, out_shape in (((12, 12), (4, 6)),
                                        ((6, 15), (3, 5)),
                                        ((8, 6), (8, 2)),
                                        ((9, 4), (3, 4))):
                with self.subTest(dtype=dtype, in_shape=in_shape, out_shape=out_shape):
                    a = np.arange(np.prod(in_shape)).reshape(in_shape).astype(dtype)
                    M, N = in_shape
                    m, n = out_shape
                    expected = a.astype(np.float64).reshape((m, M // m, n, N // n)).mean(3).mean(1)
                    out = rebin2d(xp.asarray(a), out_shape, xp=xp)
                    self.assertEqual(out.shape, out_shape)
                    np.testing.assert_allclose(cpuArray(out), expected, rtol=1e-6)

    @cpu_and_gpu
    def test_same_shape(self, target_device_idx, xp):
        a = xp.ones((4, 6))
        self.assertIs(rebin2d(a, (4, 6), xp=xp), a)

    @cpu_and_gpu
    def test_unsupported_shapes(self, target_device_idx, xp):
        a = xp.ones((6, 8))
        cases = {(4, 4): 'non-integer',        # 6 -> 4
                 (12, 16): 'Upsampling with sample=False',
                 (12, 4): 'different axes',
                 (3, 16): 'different axes'}
        for shape, message in cases.items():
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, message):
                    rebin2d(a, shape, xp=xp)


if __name__ == '__main__':
    unittest.main()
