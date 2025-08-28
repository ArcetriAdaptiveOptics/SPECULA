
import specula
specula.init(0)  # Default target device

import tempfile
import os
import gc
import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.electric_field_combinator import ElectricFieldCombinator

from test.specula_testlib import cpu_and_gpu

class TestElectricField(unittest.TestCase):

    @cpu_and_gpu
    def test_reset_does_not_reallocate(self, target_device_idx, xp):

        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)

        id_field_before = id(ef.field)        

        ef.reset()

        id_field_after = id(ef.field)

        assert id_field_before == id_field_after

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):

        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)

        id_field_before = id(ef.field)        

        ef.set_value([xp.ones(100).reshape(10,10), xp.zeros(100).reshape(10,10)])

        id_field_after = id(ef.field)
        
        assert id_field_before == id_field_after
        

    @cpu_and_gpu
    def test_ef_combinator(self, target_device_idx, xp):
        pixel_pitch = 0.1
        pixel_pupil = 10
        ef1 = ElectricField(pixel_pupil,pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef2 = ElectricField(pixel_pupil,pixel_pupil, pixel_pitch, S0=2, target_device_idx=target_device_idx)
        A1 = xp.ones((pixel_pupil, pixel_pupil))
        ef1.A = A1
        ef1.phaseInNm = 1 * xp.ones((pixel_pupil, pixel_pupil))
        
        A2 = xp.ones((pixel_pupil, pixel_pupil))
        A2[0, 0] = 0
        A2[9, 9] = 0        
        ef2.A = A2
        ef2.phaseInNm = 3 * xp.ones((pixel_pupil, pixel_pupil))

        ef_combinator = ElectricFieldCombinator(
            target_device_idx=target_device_idx
        )

        ef_combinator.inputs['in_ef1'].set(ef1)
        ef_combinator.inputs['in_ef2'].set(ef2)

        t = 1
        ef1.generation_time = t
        ef2.generation_time = t

        ef_combinator.check_ready(t)
        ef_combinator.setup()
        ef_combinator.trigger()
        ef_combinator.post_trigger()

        out_ef = ef_combinator.outputs['out_ef']

        assert np.allclose(out_ef.A, ef1.A * ef2.A)
        assert np.allclose(out_ef.phaseInNm, ef1.phaseInNm + ef2.phaseInNm)
        assert np.allclose(out_ef.S0, ef1.S0 + ef2.S0)

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        S0 = 1.23

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=S0, target_device_idx=target_device_idx)
        ef.A = xp.arange(pixel_pupil * pixel_pupil, dtype=ef.dtype).reshape(pixel_pupil, pixel_pupil)
        ef.phaseInNm = xp.arange(pixel_pupil * pixel_pupil, dtype=ef.dtype).reshape(pixel_pupil, pixel_pupil) * 0.5

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "ef_test.fits")
            ef.save(filename)

            # Restore from file
            ef2 = ElectricField.restore(filename, target_device_idx=target_device_idx)

            # Check that the restored object has the data as expected
            assert np.allclose(cpuArray(ef.A), cpuArray(ef2.A))
            assert np.allclose(cpuArray(ef.phaseInNm), cpuArray(ef2.phaseInNm))
            assert ef.pixel_pitch == ef2.pixel_pitch
            assert ef.S0 == ef2.S0

            # Force cleanup for Windows
            del ef2
            gc.collect()
            
    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        S0 = 1.23
        shape = (pixel_pupil, pixel_pupil)
        amp = xp.ones(shape)
        phase = xp.arange(pixel_pupil**2).reshape(shape) * 0.5

        ef = ElectricField(shape[0], shape[1], pixel_pitch, S0=S0, target_device_idx=target_device_idx)
        ef.set_value([amp, phase])

        assert np.allclose(cpuArray(ef.A), cpuArray(amp))
        assert np.allclose(cpuArray(ef.phaseInNm), cpuArray(phase))
        assert ef.A.dtype == ef.dtype
        assert ef.phaseInNm.dtype == ef.dtype
        
    @cpu_and_gpu
    def test_get_value(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        S0 = 1.23
        shape = (pixel_pupil, pixel_pupil)
        amp = xp.ones(shape)
        phase = xp.arange(pixel_pupil**2).reshape(shape) * 0.5

        ef = ElectricField(shape[0], shape[1], pixel_pitch, S0=S0, target_device_idx=target_device_idx)
        ef.set_value([amp, phase])

        retrieved_amp, retrieved_phase = ef.get_value()

        assert np.allclose(cpuArray(retrieved_amp), cpuArray(amp))
        assert np.allclose(cpuArray(retrieved_phase), cpuArray(phase))
        assert retrieved_amp.dtype == ef.dtype
        assert retrieved_phase.dtype == ef.dtype

    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        S0 = 1.23
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=S0, target_device_idx=target_device_idx)

        hdr = ef.get_fits_header()

        assert hdr['VERSION'] == 1
        assert hdr['OBJ_TYPE'] == 'ElectricField'
        assert hdr['DIMX'] == pixel_pupil
        assert hdr['DIMY'] == pixel_pupil
        assert hdr['PIXPITCH'] == pixel_pitch
        assert hdr['S0'] == S0        
        
    @cpu_and_gpu
    def test_with_invalid_shape(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        S0 = 1.23

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=S0, target_device_idx=target_device_idx)

        # invalid phase shape
        with self.assertRaises(AssertionError):
            ef.set_value([xp.ones((10, 10)), xp.zeros((5, 5))])
        
        # invalid amplitude shape
        with self.assertRaises(AssertionError):
            ef.set_value([xp.ones((5, 5)), xp.zeros((10, 10))])

    @cpu_and_gpu
    def test_float(self, target_device_idx, xp):
        '''Test that precision=1 results in a single-precision ef'''

        pixel_pupil = 10
        pixel_pitch = 0.1
        
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch,
                      target_device_idx=target_device_idx, precision=1)

        assert ef.field.dtype == np.float32
        assert ef.ef_at_lambda(500.0).dtype == np.complex64

    @cpu_and_gpu
    def test_double(self, target_device_idx, xp):
        '''Test that precision=0 results in a double-precision ef'''

        pixel_pupil = 10
        pixel_pitch = 0.1
        
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch,
                      target_device_idx=target_device_idx, precision=0)

        assert ef.field.dtype == np.float64
        assert ef.ef_at_lambda(500.0).dtype == np.complex128
        
    @cpu_and_gpu
    def test_ef_resize(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch,
                      target_device_idx=target_device_idx)

        # Resize to a larger size
        new_dimx = 20
        new_dimy = 20
        ef.resize(dimx=new_dimx, dimy=new_dimy)

        assert ef.field.shape == (2, new_dimx, new_dimy)
        assert ef.A.shape == (new_dimx, new_dimy)
        assert ef.phaseInNm.shape == (new_dimx, new_dimy)
        assert ef.pixel_pitch == pixel_pitch  # Unchanged by resize

    @cpu_and_gpu
    def test_ef_resize_with_pitch(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch,
                      target_device_idx=target_device_idx)

        # Resize with a new pixel pitch
        new_dimx = 15
        new_dimy = 15
        new_pixel_pitch = 0.2
        ef.resize(dimx=new_dimx, dimy=new_dimy, pitch=new_pixel_pitch)

        assert ef.field.shape == (2, new_dimx, new_dimy)
        assert ef.A.shape == (new_dimx, new_dimy)
        assert ef.phaseInNm.shape == (new_dimx, new_dimy)
        assert ef.pixel_pitch == new_pixel_pitch

    @cpu_and_gpu
    def test_area(self, target_device_idx, xp):
        """
        Test that ElectricField.area() returns the correct total area based on pixel_pitch.
        """
        ef = ElectricField(4, 5, pixel_pitch=0.2, target_device_idx=target_device_idx)
        expected_area = ef.field[0].size * (ef.pixel_pitch ** 2)
        self.assertAlmostEqual(ef.area(), expected_area)

    @cpu_and_gpu
    def test_square_modulus(self, target_device_idx, xp):
        """
        Test that ElectricField.square_modulus() computes |E|^2 correctly.
        """
        ef = ElectricField(3, 3, pixel_pitch=1.0, target_device_idx=target_device_idx)
        wavelength = 500.0  # nm

        # Set amplitude = 2 everywhere, zero phase
        ef.field[0] = xp.full((3, 3), 2.0, dtype=ef.dtype)
        ef.field[1] = xp.zeros((3, 3), dtype=ef.dtype)

        # Expected intensity = amplitude^2 = 4
        expected = xp.full((3, 3), 4.0, dtype=ef.dtype)
        result = ef.square_modulus(wavelength)

        self.assertTrue(xp.allclose(result, expected))

    @cpu_and_gpu
    def test_sub_ef_with_indices(self, target_device_idx, xp):
        """
        Test ElectricField.sub_ef() when extracting a subregion using explicit indices.
        """
        ef = ElectricField(6, 6, pixel_pitch=0.5, target_device_idx=target_device_idx)
        ef.field[0] = xp.arange(36, dtype=ef.dtype).reshape(6, 6)
        ef.field[1] = ef.field[0] * 2

        sub_ef = ef.sub_ef(xfrom=2, xto=5, yfrom=1, yto=4)
        expected_amplitude = ef.field[0, 2:5, 1:4]
        expected_phase = ef.field[1, 2:5, 1:4]

        self.assertTrue(xp.allclose(sub_ef.field[0], expected_amplitude))
        self.assertTrue(xp.allclose(sub_ef.field[1], expected_phase))
        self.assertEqual(sub_ef.pixel_pitch, ef.pixel_pitch)
        self.assertEqual(sub_ef.S0, ef.S0)

    @cpu_and_gpu
    def test_sub_ef_with_flat_indices(self, target_device_idx, xp):
        """
        Test ElectricField.sub_ef() when using a flat array of indices.
        """
        ef = ElectricField(4, 4, pixel_pitch=0.1, target_device_idx=target_device_idx)
        ef.field[0] = xp.arange(16, dtype=ef.dtype).reshape(4, 4)
        ef.field[1] = ef.field[0] * 2

        # Select three arbitrary points in flat indexing
        idx = xp.array([0, 5, 10])
        sub_ef = ef.sub_ef(idx=idx)

        # Bounds should match min/max idx selection
        min_x, max_x = xp.min(xp.unravel_index(idx, ef.field[0].shape)[0]), xp.max(xp.unravel_index(idx, ef.field[0].shape)[0])
        min_y, max_y = xp.min(xp.unravel_index(idx, ef.field[0].shape)[1]), xp.max(xp.unravel_index(idx, ef.field[0].shape)[1])

        expected_amplitude = ef.field[0, min_x:max_x, min_y:max_y]
        expected_phase = ef.field[1, min_x:max_x, min_y:max_y]

        self.assertTrue(xp.allclose(sub_ef.field[0], expected_amplitude))
        self.assertTrue(xp.allclose(sub_ef.field[1], expected_phase))

    @cpu_and_gpu
    def test_check_other_success(self, target_device_idx, xp):
        """
        Test ElectricField.checkOther() when ef2 has matching size.
        """
        ef1 = ElectricField(4, 4, pixel_pitch=1.0, target_device_idx=target_device_idx)
        ef2 = ElectricField(4, 4, pixel_pitch=1.0, target_device_idx=target_device_idx)

        subrect = ef1.checkOther(ef2)
        self.assertEqual(subrect, [0, 0])

    @cpu_and_gpu
    def test_check_other_invalid_type(self, target_device_idx, xp):
        """
        Test ElectricField.checkOther() raises ValueError if ef2 is not an ElectricField.
        """
        ef1 = ElectricField(4, 4, pixel_pitch=1.0, target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            ef1.checkOther("not an ElectricField")

    @cpu_and_gpu
    def test_check_other_size_mismatch(self, target_device_idx, xp):
        """
        Test ElectricField.checkOther() raises ValueError when ef2 has incompatible size.
        """
        ef1 = ElectricField(4, 4, pixel_pitch=1.0, target_device_idx=target_device_idx)
        ef2 = ElectricField(2, 2, pixel_pitch=1.0, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            ef1.checkOther(ef2)
