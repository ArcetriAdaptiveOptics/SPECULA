import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.graph_matmul import left_mul, matmul, matvec, vecmat
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


class TestMatmul(unittest.TestCase):

    CASES = {
        'matrix @ vector': ((30, 50), (50,), False),
        'transposed matrix @ vector': ((50, 30), (50,), 'a'),
        'vector @ matrix': ((50,), (50, 3000), False),
        'vectors @ matrix': ((3, 50), (50, 3000), False),
        'vector @ transposed matrix': ((50,), (60, 50), True),
        'vectors @ transposed matrix': ((3, 50), (60, 50), True),
    }

    def operands(self, xp, dtype, a_shape, b_shape, transposed):
        rng = np.random.default_rng(7)
        a = xp.asarray(rng.standard_normal(a_shape).astype(dtype))
        b = xp.asarray(rng.standard_normal(b_shape).astype(dtype))
        if transposed == 'a':
            return a.T, b
        return a, (b.T if transposed else b)

    @cpu_and_gpu
    def test_matches_matmul(self, target_device_idx, xp):
        for name, (a_shape, b_shape, transposed) in self.CASES.items():
            with self.subTest(name):
                a, b = self.operands(xp, np.float32, a_shape, b_shape, transposed)
                np.testing.assert_allclose(cpuArray(matmul(a, b, xp)), cpuArray(a @ b), rtol=1e-4, atol=1e-4)

    @cpu_and_gpu
    def test_vecmat(self, target_device_idx, xp):
        for dtype, tol in ((np.float64, 1e-10), (np.float32, 1e-4)):
            for a_shape in ((70,), (2, 70)):
                with self.subTest(dtype=dtype, a_shape=a_shape):
                    a, b = self.operands(xp, dtype, a_shape, (70, 1000), False)
                    np.testing.assert_allclose(cpuArray(vecmat(a, b, xp)), cpuArray(a @ b), rtol=tol, atol=tol)

    @cpu_and_gpu
    def test_matvec(self, target_device_idx, xp):
        for dtype, tol in ((np.float64, 1e-10), (np.float32, 1e-4)):
            for v_shape in ((70,), (2, 70)):
                with self.subTest(dtype=dtype, v_shape=v_shape):
                    a, v = self.operands(xp, dtype, (300, 70), v_shape, False)
                    expected = a @ v if v.ndim == 1 else v @ a.T
                    np.testing.assert_allclose(cpuArray(matvec(a, v, xp)), cpuArray(expected), rtol=tol, atol=tol)

    @cpu_and_gpu
    def test_captured_in_cuda_graph(self, target_device_idx, xp):
        """Inside a capture, cuBLAS is replaced by kernels that can be captured"""
        if xp is np:
            self.skipTest('CUDA graphs are GPU only')
        stream = xp.cuda.Stream(non_blocking=True)
        for name, (a_shape, b_shape, transposed) in self.CASES.items():
            with self.subTest(name):
                a, b = self.operands(xp, np.float32, a_shape, b_shape, transposed)
                matmul(a, b, xp)    # compiles the kernels outside the capture
                with stream:
                    stream.begin_capture()
                    out = matmul(a, b, xp)
                    graph = stream.end_capture()
                a *= 2              # the graph uses the current values
                graph.launch(stream=stream)
                stream.synchronize()
                np.testing.assert_allclose(cpuArray(out), cpuArray(a @ b), rtol=1e-4, atol=1e-4)

