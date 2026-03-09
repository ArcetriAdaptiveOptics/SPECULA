import unittest
import specula
specula.init(0)  # Default target device

from specula import np, cpuArray
from specula.data_objects.finite_phase_screen import FinitePhaseScreen

from test.specula_testlib import cpu_and_gpu


class TestFinitePhaseScreen(unittest.TestCase):

    @cpu_and_gpu
    def test_screen_stored_on_correct_device(self, target_device_idx, xp):
        """Screen data should be stored on the correct device (CPU or GPU)."""
        screen = np.random.rand(128, 512).astype(np.float32)
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)

        self.assertEqual(type(fps.screen), type(xp.zeros(1, dtype=xp.float32)))
        self.assertEqual(fps.height, 128)
        self.assertEqual(fps.width, 512)

    @cpu_and_gpu
    def test_extract_phase_output_size(self, target_device_idx, xp):
        """Extracted patch must have the requested output_size."""
        screen = xp.ones((256, 1024), dtype=xp.float32)
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)

        for output_size in [64, 128, 200]:
            patch = fps.extract_phase(shift_step=0, angle_deg=0.0, output_size=output_size)
            self.assertEqual(patch.shape, (output_size, output_size))

    @cpu_and_gpu
    def test_extract_phase_zero_angle_is_pure_shift(self, target_device_idx, xp):
        """With angle=0 and a known screen, the patch must be a contiguous slice."""
        # Build a screen where each column has a unique constant value
        width, height = 512, 256
        col_values = xp.arange(width, dtype=xp.float32)
        screen = xp.tile(col_values, (height, 1))
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)

        output_size = 64
        shift = 10
        patch = cpuArray(fps.extract_phase(shift_step=shift, angle_deg=0.0,
                                           output_size=output_size))

        # With angle=0 and order=1 bilinear rotation the patch should be unmodified
        # columns of the patch should correspond to shifted column indices
        expected_col = shift % (width - output_size)
        np.testing.assert_allclose(patch[:, 0], expected_col, atol=1e-6)

    @cpu_and_gpu
    def test_shift_wraps_with_modulo(self, target_device_idx, xp):
        """Large shift_step values must wrap around via modulo."""
        screen = np.random.rand(256, 512).astype(np.float32)
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)
        output_size = 128
        max_shift = fps.width - output_size  # 384

        patch_a = cpuArray(fps.extract_phase(shift_step=5, angle_deg=0.0,
                                             output_size=output_size))
        patch_b = cpuArray(fps.extract_phase(shift_step=5 + max_shift, angle_deg=0.0,
                                             output_size=output_size))

        np.testing.assert_array_equal(patch_a, patch_b)

    @cpu_and_gpu
    def test_different_shifts_produce_different_patches(self, target_device_idx, xp):
        """Different shift values must generally produce different patches."""
        rng = np.random.default_rng(42)
        screen = rng.standard_normal((256, 512))
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)
        output_size = 64

        patch_a = cpuArray(fps.extract_phase(shift_step=0,  angle_deg=0.0,
                                             output_size=output_size))
        patch_b = cpuArray(fps.extract_phase(shift_step=50, angle_deg=0.0,
                                             output_size=output_size))

        self.assertFalse(np.allclose(patch_a, patch_b),
                         "Different shifts should produce different patches")

    @cpu_and_gpu
    def test_nonzero_angle_rotates_patch(self, target_device_idx, xp):
        """A nonzero rotation angle must change the extracted patch."""
        rng = np.random.default_rng(7)
        screen = rng.standard_normal((256, 512))
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)
        output_size = 64

        patch_0   = cpuArray(fps.extract_phase(shift_step=0, angle_deg=0.0,
                                               output_size=output_size))
        patch_45  = cpuArray(fps.extract_phase(shift_step=0, angle_deg=45.0,
                                               output_size=output_size))

        self.assertFalse(np.allclose(patch_0, patch_45),
                         "Different angles should produce different patches")

    @cpu_and_gpu
    def test_width_smaller_than_output_size_raises(self, target_device_idx, xp):
        """extract_phase must raise ValueError when screen width <= output_size."""
        screen = np.ones((64, 64), dtype=np.float32)
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            fps.extract_phase(shift_step=0, angle_deg=0.0, output_size=64)

    @cpu_and_gpu
    def test_extract_patch_does_not_modify_screen(self, target_device_idx, xp):
        """Calling extract_phase must not alter the stored screen."""
        rng = np.random.default_rng(99)
        screen = rng.standard_normal((256, 512))
        fps = FinitePhaseScreen(screen, target_device_idx=target_device_idx)
        original = cpuArray(fps.screen.copy())

        fps.extract_phase(shift_step=10, angle_deg=30.0, output_size=64)

        np.testing.assert_array_equal(cpuArray(fps.screen), original)
