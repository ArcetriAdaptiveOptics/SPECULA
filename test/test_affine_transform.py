import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray, np
from specula.lib.affine_transform import affine_transform

from test.specula_testlib import cpu_and_gpu


def _rotation(deg, n):
    """Affine matrix of a rotation by deg around the center of an n x n array"""
    rad = np.radians(deg)
    a, b = np.round(np.cos(rad), 15), np.round(np.sin(rad), 15)
    c = (n - 1) / 2
    return np.array([[a, b, c * (1 - a - b)], [-b, a, c * (1 - a + b)]])


class TestAffineTransform(unittest.TestCase):

    def _check(self, data, matrix, out_shape, xp, dtype):
        from scipy.ndimage import affine_transform as scipy_affine_transform
        data = np.asarray(data, dtype=dtype)
        matrix = np.asarray(matrix, dtype=dtype)
        expected = scipy_affine_transform(data, matrix, output_shape=out_shape, order=1)
        output = xp.zeros(out_shape, dtype=dtype)
        affine_transform(xp.asarray(data), xp.asarray(matrix), output, xp=xp)
        rtol = 1e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(cpuArray(output), expected, rtol=rtol,
                                   atol=rtol * np.abs(expected).max())

    @cpu_and_gpu
    def test_matches_scipy(self, target_device_idx, xp):
        rng = np.random.default_rng(1)
        for dtype in [np.float32, np.float64]:
            screen = rng.standard_normal((40, 300))
            window = rng.standard_normal((40, 40))
            # Window extraction with fractional and integer shifts
            for shift in [0.0, 3.25, 17.0, 250.9]:
                self._check(screen, [[1, 0, 0], [0, 1, shift]], (40, 40), xp, dtype)
            # rot90() and flips, also with fractional offsets on both axes
            for matrix in [[[0, 1, 0], [-1, 0, 39]], [[-1, 0, 39], [0, -1, 39]],
                           [[0, -1, 39], [1, 0, 0]], [[1, 0, 0], [0, -1, 39]],
                           [[0, 1, 0.5], [1, 0, 1.25]], [[-1, 0, 38.5], [0, 1, 0.75]]]:
                self._check(window, matrix, (40, 40), xp, dtype)
            # Samples outside the input (zero there, as ndimage 'constant' mode)
            for matrix in [[[1, 0, -3.5], [0, 1, 2.25]], [[0, -1, 41.5], [1, 0, -0.5]]]:
                self._check(window, matrix, (40, 40), xp, dtype)
            # Fractional rotations, and a non-square input and output
            for deg in [0, 90, 180, -90, 33.3, -212.7, 359.9]:
                self._check(window, _rotation(deg, 40), (40, 40), xp, dtype)
            self._check(screen[:, :60], [[0.9, 0.2, 1.0], [-0.1, 1.1, 3.0]], (30, 50), xp, dtype)

    def test_permutation_is_exact(self):
        """Window shifts and rot90() use strided views: same values as slicing"""
        rng = np.random.default_rng(2)
        screen = rng.standard_normal((40, 300)).astype(np.float32)
        output = np.zeros((40, 40), dtype=np.float32)
        affine_transform(screen, np.array([[1, 0, 0], [0, 1, 17.25]], dtype=np.float32),
                         output, xp=np)
        r = np.float32(0.25)
        np.testing.assert_array_equal(output, (1 - r) * screen[:, 17:57] + r * screen[:, 18:58])
        window = output.copy()
        affine_transform(window, np.array([[0, 1, 0], [-1, 0, 39]], dtype=np.float32),
                         output, xp=np)
        np.testing.assert_array_equal(output, np.rot90(window))


if __name__ == '__main__':
    unittest.main()
