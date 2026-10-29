import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.graph_matmul import left_mul
from test.specula_testlib import cpu_and_gpu


class TestLeftMul(unittest.TestCase):

    @cpu_and_gpu
    def test_matches_matmul(self, target_device_idx, xp):
        """a @ x[b] for a batch, with sizes that are not multiples of the 16 x 16 tiles,
        also with a strided (transposed) x, in single and double precision"""
        rng = np.random.default_rng(5)
        nb, r, n, m = 3, 7, 37, 21
        a = rng.standard_normal((r, n))
        x = rng.standard_normal((nb, n, m))
        expected = a[None] @ x
        for dtype, tol in ((np.float64, 1e-12), (np.float32, 1e-5)):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    xd = xp.asarray(x.swapaxes(1, 2).astype(dtype)).swapaxes(1, 2) if strided \
                        else xp.asarray(x.astype(dtype))
                    out = xp.zeros((nb, r, m), dtype=dtype)
                    left_mul(xp.asarray(a.astype(dtype)), xd, out, xp)
                    np.testing.assert_allclose(cpuArray(out), expected, rtol=tol, atol=tol)

    @cpu_and_gpu
    def test_captured_in_cuda_graph(self, target_device_idx, xp):
        """The reason for this module: the product can be captured in a CUDA graph"""
        if xp is np:
            self.skipTest('CUDA graphs are GPU only')
        rng = np.random.default_rng(6)
        a = xp.asarray(rng.standard_normal((5, 40)))
        x = xp.asarray(rng.standard_normal((4, 40, 30)))
        out = xp.zeros((4, 5, 30))
        stream = xp.cuda.Stream(non_blocking=True)
        with stream:
            left_mul(a, x, out, xp)    # compiles the kernel outside the capture
            stream.begin_capture()
            left_mul(a, x, out, xp)
            graph = stream.end_capture()
            out[:] = 0
            x *= 2
            graph.launch(stream)
            stream.synchronize()
        np.testing.assert_allclose(cpuArray(out), cpuArray(a[None] @ x), rtol=1e-12, atol=1e-12)

    @cpu_and_gpu
    def test_raw_kernel_rejects_unsupported_layouts(self, target_device_idx, xp):
        """The GPU kernel reads a row-major and uses the dtype of the output"""
        if xp is np:
            self.skipTest('the checks apply to the GPU kernel only')
        a = xp.ones((3, 8))
        x = xp.ones((2, 8, 8))
        out = xp.zeros((2, 3, 8))
        with self.assertRaises(ValueError):
            left_mul(xp.ones((8, 3)).T, x, out, xp)
        with self.assertRaises(ValueError):
            left_mul(a, x.astype(xp.float32), out, xp)
        with self.assertRaises(ValueError):
            left_mul(a, x, xp.zeros((2, 8, 3)).swapaxes(1, 2), xp)


if __name__ == '__main__':
    unittest.main()
