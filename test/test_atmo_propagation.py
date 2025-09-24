import specula
specula.init(0)  # Default target device

import os
import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.source import Source
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.layer import Layer
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoPropagation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup test data path"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'calib', 'pupilstop')
        cls.pupil_fits_file = os.path.join(cls.test_data_dir, 'EELT480pp0.0803m_obs0.283_spider2023.fits')

    @cpu_and_gpu
    def test_propagation_without_magnification(self, target_device_idx, xp):
        """Test propagation without magnification - should extract center region"""

        # Setup simulation parameters
        pixel_pupil = 240  # Half the size of the FITS file (480x480)
        pixel_pitch = 0.0803  # From FITS filename
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        # Load pupil stop from FITS
        pupilstop = Pupilstop.restore(self.pupil_fits_file, target_device_idx=target_device_idx)

        # Resize pupil to be larger than simulation pupil (480 -> keep original size)
        # This will test the center extraction

        # Create atmospheric layer at ground level (no magnification)
        layer = Layer(
            dimx=pupilstop.A.shape[0],  # 480
            dimy=pupilstop.A.shape[1],  # 480
            pixel_pitch=pixel_pitch,
            height=0.0,  # Ground layer
            magnification=1.0,  # No magnification
            target_device_idx=target_device_idx
        )

        # Set layer amplitude to pupil pattern
        layer.A = pupilstop.A.copy()
        layer.phaseInNm = xp.zeros_like(layer.A)
        layer.generation_time = 1

        # Create source
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        # Create propagation object
        prop = AtmoPropagation(
            simul_params,
            source_dict={'on_axis': on_axis_source},
            target_device_idx=target_device_idx
        )
        
        # Connect inputs
        prop.inputs['atmo_layer_list'].set([])  # No atmo layers
        prop.inputs['common_layer_list'].set([layer])  # Only ground layer

        # Setup and run
        prop.setup()
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()

        # Get output
        output_ef = prop.outputs['out_on_axis_ef']

        # Expected result: center 240x240 region of the 480x480 pupil
        expected_topleft = [(480 - 240) // 2, (480 - 240) // 2]  # [120, 120]
        expected_region = cpuArray(pupilstop.A[
            expected_topleft[0]:expected_topleft[0] + 240,
            expected_topleft[1]:expected_topleft[1] + 240
        ])

        # Check that output matches expected center extraction
        output_amplitude = cpuArray(output_ef.A)

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.imshow(cpuArray(layer.A), cmap='gray', vmin=0, vmax=1, origin='lower')
            plt.colorbar()
            plt.title('Layer Amplitude with Bright Square')
            plt.figure(figsize=(6,6))
            plt.imshow(output_amplitude, cmap='gray', vmin=0, vmax=1, origin='lower')
            plt.colorbar()
            plt.title('Output Amplitude without Magnification')
            plt.show()

        assert output_amplitude.shape == (240, 240), f"Expected (240, 240), got {output_amplitude.shape}"
        assert np.allclose(output_amplitude, expected_region), "Center extraction doesn't match expected region"

    @cpu_and_gpu
    def test_propagation_with_magnification(self, target_device_idx, xp):
        """Test propagation with magnification - should use interpolation"""

        pixel_pupil = 240
        pixel_pitch = 0.0803
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        # Create layer with magnification
        magnification = 2.0  # Double magnification
        layer = Layer(
            dimx=pixel_pupil,
            dimy=pixel_pupil,
            pixel_pitch=pixel_pitch,
            height=1000.0,  # Elevated layer
            magnification=magnification,
            target_device_idx=target_device_idx
        )

        # Create a test pattern that's easy to verify
        # Put a bright square in the middle
        layer.A = xp.zeros_like(layer.A)
        layer.A[int(pixel_pupil/2)-50:int(pixel_pupil/2)+50, int(pixel_pupil/2)-50:int(pixel_pupil/2)+50] = 1.0 
        layer.phaseInNm = xp.zeros_like(layer.A)
        layer.generation_time = 1

        # Create source
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        # Create propagation
        prop = AtmoPropagation(
            simul_params,
            source_dict={'on_axis': on_axis_source},
            target_device_idx=target_device_idx
        )

        prop.inputs['atmo_layer_list'].set([layer])
        prop.inputs['common_layer_list'].set([])

        # Setup and run
        prop.setup()
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()

        # Get output
        output_ef = prop.outputs['out_on_axis_ef']
        output_amplitude = cpuArray(output_ef.A)

        # With magnification, should use interpolation (not direct extraction)
        # The bright square should be visible but interpolated
        assert output_amplitude.shape == (240, 240)
        # total of output amplitude must be approx magnification**2 times input amplitude
        assert np.isclose(np.sum(output_amplitude), np.sum(layer.A) * magnification**2, rtol=0.001), \
            f"Output sum {np.sum(output_amplitude)} should be approx {np.sum(layer.A) * magnification**2}"

    @cpu_and_gpu
    def test_quarter_array_extraction(self, target_device_idx, xp):
        """Test geometric setup that extracts a quarter of the array"""

        pixel_pupil = 120  # Quarter of 480
        pixel_pitch = 0.0803
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        dim_layer = pixel_pupil * 2  # 240 to allow quarter extraction
        height = 1000.0

        # Create layer with specific shift to get quarter extraction
        layer = Layer(
            dimx=dim_layer, dimy=dim_layer,
            pixel_pitch=pixel_pitch,
            height=height,
            #shiftXYinPixel=(90.0, 90.0),  # Shift to offset the center
            target_device_idx=target_device_idx
        )

        # Create a checkerboard pattern for easy verification
        layer.A = xp.zeros((dim_layer, dim_layer))
        layer.A[int(dim_layer/2):, int(dim_layer/2):] = 1.0  # one quarter bright
        layer.phaseInNm = xp.zeros_like(layer.A)
        layer.phaseInNm[int(dim_layer/2):, int(dim_layer/2):] = 2.0  # one quarter bright
        layer.generation_time = 1

        # Off-axis source to create geometric offset
        radius = np.sqrt(2) * np.arctan((pixel_pupil * pixel_pitch / 2) / height) * (180.0 / np.pi) * 3600  # in arcsec
        off_axis_source = Source(polar_coordinates=[radius, 45.0], magnitude=8, wavelengthInNm=750)

        prop = AtmoPropagation(
            simul_params,
            source_dict={'off_axis': off_axis_source},
            target_device_idx=target_device_idx
        )

        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])

        prop.setup()
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()

        output_ef = prop.outputs['out_off_axis_ef']
        output_amplitude = cpuArray(output_ef.A)
        output_phase = cpuArray(output_ef.phaseInNm)

        # output amplitude must be 1
        assert np.max(output_amplitude) > 0.99, f"Max amplitude {np.max(output_amplitude)} should be > 0.99"
        assert np.min(output_amplitude) < 1.01, f"Min amplitude {np.min(output_amplitude)} should be < 1.01"
        assert np.isclose(np.mean(output_amplitude), 1.0, rtol=0.01), \
            f"Mean amplitude {np.mean(output_amplitude)} should be approx 1.0"
        assert np.max(output_phase) > 1.99, f"Max phase {np.max(output_phase)} should be > 1.99"
        assert np.min(output_phase) < 2.01, f"Min phase {np.min(output_phase)} should be < 2.01"
        assert np.isclose(np.mean(output_phase), 2.0, rtol=0.01), \
            f"Mean phase {np.mean(output_phase)} should be approx 1.0"

    @cpu_and_gpu
    def test_interpolation_artifacts_correction(self, target_device_idx, xp):
        """Test that phase correction for interpolation artifacts works"""

        pixel_pupil = 200
        pixel_pitch = 0.0803
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        # Create layer with holes (zero amplitude regions)
        layer = Layer(
            dimx=400, dimy=400,
            pixel_pitch=pixel_pitch,
            height=5000.0,
            magnification=1.5,  # Will trigger interpolation
            target_device_idx=target_device_idx
        )

        # Create amplitude with holes
        layer.A = xp.ones((400, 400))
        layer.A[100:120, 100:120] = 0  # Create a hole
        layer.A[200:250, 200:250] = 0  # Another hole

        # Create phase with known values
        layer.phaseInNm = xp.ones((400, 400)) * 100.0  # Base phase
        layer.phaseInNm[50:150, 50:150] = 200.0  # Different phase region

        layer.generation_time = 1

        # Store original phase in holes
        original_hole_phase = cpuArray(layer.phaseInNm[layer.A == 0])

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        prop = AtmoPropagation(
            simul_params,
            source_dict={'on_axis': on_axis_source},
            target_device_idx=target_device_idx
        )

        prop.inputs['atmo_layer_list'].set([layer])
        prop.inputs['common_layer_list'].set([])

        prop.setup()

        # Run prepare_trigger to apply phase correction
        prop.prepare_trigger(1)

        # Check that holes have been filled with local mean
        filled_hole_phase = cpuArray(layer.phaseInNm[cpuArray(layer.A) == 0])

        # Holes should no longer have original values
        assert not np.allclose(filled_hole_phase, original_hole_phase), \
            "Phase in holes should be modified"

        # Continue with propagation
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()

        output_ef = prop.outputs['out_on_axis_ef']
        assert output_ef.A.shape == (200, 200)

    @cpu_and_gpu
    def test_layer_shiftXYinPixel(self, target_device_idx, xp):
        """Test that layer shiftXYinPixel works correctly"""
        pixel_pupil = 100
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        dim_layer = 120  # Larger than pupil to allow shifting

        # Layer with shift of 20 pixels in x and 10 in y
        layer = Layer(
            dimx=dim_layer, dimy=dim_layer,
            pixel_pitch=pixel_pitch,
            height=0.0,
            shiftXYinPixel=(20.0, 10.0),
            target_device_idx=target_device_idx
        )
        layer.A = xp.zeros((dim_layer, dim_layer))
        layer.A[dim_layer//2-30:dim_layer//2+30, dim_layer//2-30:dim_layer//2+30] = xp.ones((60, 60))
        layer.phaseInNm = xp.zeros((dim_layer, dim_layer))
        layer.generation_time = 1

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        prop = AtmoPropagation(simul_params, source_dict={'on_axis': source}, target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])
        prop.setup()
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()

        output_ef = prop.outputs['out_on_axis_ef']
        output_amplitude = cpuArray(output_ef.A)

        # The bright square should be shifted by (20,10) pixels in the output
        expected_amplitude = (np.roll(np.roll(cpuArray(layer.A), 20, axis=1), 10, axis=0))
        expected_amplitude = expected_amplitude[dim_layer//2 - pixel_pupil//2:dim_layer//2 + pixel_pupil//2, \
                                                dim_layer//2 - pixel_pupil//2:dim_layer//2 + pixel_pupil//2]
        diff = output_amplitude - expected_amplitude

        max_diff = np.max(np.abs(diff))
        assert max_diff < 1e-5, f"Max difference after shift is {max_diff}, should be < 1e-5"
        
    @cpu_and_gpu
    def test_layer_rotInDeg(self, target_device_idx, xp):
        """Test that layer rotInDeg works correctly"""
        pixel_pupil = 100
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        dim_layer = 120  # Larger than pupil to allow shifting

        # Layer with shift of 20 pixels in x and 10 in y
        layer = Layer(
            dimx=dim_layer, dimy=dim_layer,
            pixel_pitch=pixel_pitch,
            height=0.0,
            rotInDeg=90.0,
            target_device_idx=target_device_idx
        )
        layer.A = np.eye(dim_layer)
        layer.phaseInNm = np.zeros((dim_layer, dim_layer))
        layer.generation_time = 1

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        prop = AtmoPropagation(simul_params, source_dict={'on_axis': source}, target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])
        prop.setup()
        prop.check_ready(1)
        prop.trigger()
        prop.post_trigger()

        output_ef = prop.outputs['out_on_axis_ef']
        output_amplitude = cpuArray(output_ef.A)

        # check that the output amplitude has a diagonal line rotated by 90deg
        expected_amplitude = np.fliplr(np.eye(pixel_pupil))
        diff = output_amplitude - expected_amplitude

        max_diff = np.max(np.abs(diff))
        assert max_diff < 0.02, f"Max difference after rotation is {max_diff}, should be < 0.02"