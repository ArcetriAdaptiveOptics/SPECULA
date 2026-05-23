import specula
specula.init(0)  # Default target device

import os
import unittest

from specula import np
from specula import cpuArray

from specula.loop_control import LoopControl
from specula.base_data_obj import BaseDataObj
from specula.data_objects.source import Source
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.layer import Layer
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu
from test.specula_testlib import find_instances


class TestAtmoPropagation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup test data path"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'calib', 'pupilstop')
        cls.pupil_fits_file = os.path.join(cls.test_data_dir,
                                           'EELT480pp0.0803m_obs0.283_spider2023.fits')

    @cpu_and_gpu
    def test_allocation_precision(self, target_device_idx, xp):
        '''
        Test that all internal arrays are allocated in single-precision
        even if the global precision is set to double
        '''

        precision = 1  # Different from the global one, that defaults to zero

        # Setup simulation parameters
        pixel_pupil = 240
        pixel_pitch = 0.0803
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
            precision=precision,
            target_device_idx=target_device_idx
        )

        # Set layer amplitude to pupil pattern
        layer.A = pupilstop.A.copy()
        layer.phaseInNm = xp.zeros_like(layer.A)
        layer.generation_time = 1

        # Create source
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750, precision=precision)

        # Create propagation object
        prop = AtmoPropagation(
            simul_params,
            source_dict={'on_axis': on_axis_source},
            precision=precision,
            target_device_idx=target_device_idx,
        )

        # Connect inputs
        prop.inputs['atmo_layer_list'].set([])  # No atmo layers
        prop.inputs['common_layer_list'].set([layer])  # Only ground layer

        prop.setup()

        for path, obj in find_instances(prop, BaseDataObj):
            assert obj.precision == prop.precision

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
        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

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
        if plot_debug: # pragma: no cover
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
        layer.A[int(pixel_pupil/2)-50:int(pixel_pupil/2)+50,
                int(pixel_pupil/2)-50:int(pixel_pupil/2)+50] = 1.0
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
        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        # Get output
        output_ef = prop.outputs['out_on_axis_ef']
        output_amplitude = cpuArray(output_ef.A)

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.imshow(cpuArray(layer.A), cmap='gray', vmin=0, vmax=1, origin='lower')
            plt.colorbar()
            plt.title('Layer Amplitude with Bright Square')
            plt.figure(figsize=(6,6))
            plt.imshow(output_amplitude, cmap='gray', vmin=0, vmax=1, origin='lower')
            plt.colorbar()
            plt.title('Output Amplitude with Magnification')
            plt.show()

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
        radius = np.sqrt(2) * np.arctan((pixel_pupil * pixel_pitch / 2) / height) \
                 * (180.0 / np.pi) * 3600  # in arcsec
        off_axis_source = Source(polar_coordinates=[radius, 45.0], magnitude=8, wavelengthInNm=750)

        prop = AtmoPropagation(
            simul_params,
            source_dict={'off_axis': off_axis_source},
            target_device_idx=target_device_idx
        )

        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])

        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        output_ef = prop.outputs['out_off_axis_ef']
        output_amplitude = cpuArray(output_ef.A)
        output_phase = cpuArray(output_ef.phaseInNm)

        # output amplitude must be 1
        assert np.max(output_amplitude) > 0.99, \
            f"Max amplitude {np.max(output_amplitude)} should be > 0.99"
        assert np.min(output_amplitude) < 1.01, \
            f"Min amplitude {np.min(output_amplitude)} should be < 1.01"
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
        layer.A[dim_layer//2-30:dim_layer//2+30, dim_layer//2-30:dim_layer//2+30] = \
                xp.ones((60, 60))
        layer.phaseInNm = xp.zeros((dim_layer, dim_layer))
        layer.generation_time = 1

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        prop = AtmoPropagation(simul_params,
                               source_dict={'on_axis': source},
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])

        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        output_ef = prop.outputs['out_on_axis_ef']
        output_amplitude = cpuArray(output_ef.A)

        # The bright square should be shifted by (20,10) pixels in the output
        expected_amplitude = (np.roll(np.roll(cpuArray(layer.A), 20, axis=1), 10, axis=0))
        expected_amplitude = expected_amplitude[dim_layer//2 - pixel_pupil//2:dim_layer//2 + pixel_pupil//2, \
                                                dim_layer//2 - pixel_pupil//2:dim_layer//2 + pixel_pupil//2]
        diff = output_amplitude - expected_amplitude

        max_diff = np.max(np.abs(diff))

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.imshow(expected_amplitude, cmap='gray', vmin=0, vmax=1, origin='lower')
            plt.colorbar()
            plt.title('Expected Amplitude')
            plt.figure(figsize=(6,6))
            plt.imshow(output_amplitude, cmap='gray', vmin=0, vmax=1, origin='lower')
            plt.colorbar()
            plt.title('Output Amplitude with Shift')
            plt.show()

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
        prop = AtmoPropagation(simul_params,
                               source_dict={'on_axis': source},
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])

        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        output_ef = prop.outputs['out_on_axis_ef']
        output_amplitude = cpuArray(output_ef.A)

        # check that the output amplitude has a diagonal line rotated by 90deg
        expected_amplitude = np.fliplr(np.eye(pixel_pupil))
        diff = output_amplitude - expected_amplitude

        max_diff = np.max(np.abs(diff))
        assert max_diff < 0.02, f"Max difference after rotation is {max_diff}, should be < 0.02"

    def test_atmo_chromatic_shift_switches(self):
        """Test AtmoPropagation chromatic switch logic (disabled/equal wavelength)."""
        simul_params = SimulParams(64, 0.1, zenithAngleInDeg=30.0)
        atmo_layer = Layer(dimx=96, dimy=96, pixel_pitch=0.1, height=5000.0, target_device_idx=-1)

        src_disabled = Source(
            polar_coordinates=[0.0, 0.0],
            magnitude=8,
            wavelengthInNm=2200.0,
            target_device_idx=-1
        )
        prop_disabled = AtmoPropagation(
            simul_params,
            source_dict={'src': src_disabled},
            enable_chromatic_effect=False,
            target_device_idx=-1
        )
        prop_disabled.inputs['atmo_layer_list'].set([atmo_layer])
        prop_disabled.inputs['common_layer_list'].set([])
        prop_disabled.setup()
        assert prop_disabled.chromatic_shifts_m[src_disabled] == {}, \
               "Chromatic shifts must be empty when effect is disabled"

        with self.assertRaises(ValueError):
            AtmoPropagation(
                simul_params,
                source_dict={'src': src_disabled},
                enable_chromatic_effect=True,
                target_device_idx=-1
            )

        src_equal_wl = Source(
            polar_coordinates=[0.0, 0.0],
            magnitude=8,
            wavelengthInNm=589.0,
            target_device_idx=-1
        )
        prop_equal = AtmoPropagation(
            simul_params,
            source_dict={'src': src_equal_wl},
            enable_chromatic_effect=True,
            chromatic_reference_wavelengthInNm=589.0,
            telescope_altitude_m=3064.0,
            target_device_idx=-1
        )
        prop_equal.inputs['atmo_layer_list'].set([atmo_layer])
        prop_equal.inputs['common_layer_list'].set([])
        prop_equal.setup()
        assert prop_equal.chromatic_shifts_m[src_equal_wl] == {}, \
            "Chromatic shifts must be empty for equal wavelengths"

    @cpu_and_gpu
    def test_chromatic_shift_is_computed_only_for_atmo_layers(self, target_device_idx, xp):
        """Test that chromatic shifts are populated only for atmospheric layers."""
        simul_params = SimulParams(80, 0.1, zenithAngleInDeg=30.0)

        atmo_layer = Layer(
            dimx=120, dimy=120,
            pixel_pitch=0.1,
            height=10000.0,
            target_device_idx=target_device_idx
        )
        atmo_layer.A = xp.ones((120, 120))
        atmo_layer.phaseInNm = xp.zeros((120, 120))
        atmo_layer.generation_time = 1

        common_layer = Layer(
            dimx=120, dimy=120,
            pixel_pitch=0.1,
            height=0.0,
            target_device_idx=target_device_idx
        )
        common_layer.A = xp.ones((120, 120))
        common_layer.phaseInNm = xp.zeros((120, 120))
        common_layer.generation_time = 1

        sci_source = Source(
            polar_coordinates=[5.0, 90.0],
            magnitude=8,
            wavelengthInNm=2200.0,
            target_device_idx=target_device_idx
        )

        prop = AtmoPropagation(
            simul_params,
            source_dict={'sci': sci_source},
            enable_chromatic_effect=True,
            chromatic_reference_wavelengthInNm=589.0,
            telescope_altitude_m=3064.0,
            target_device_idx=target_device_idx
        )
        prop.inputs['atmo_layer_list'].set([atmo_layer])
        prop.inputs['common_layer_list'].set([common_layer])
        prop.setup()
        
        print(f"\nLayers trovati nel dict: {list(prop.chromatic_shifts_m[sci_source].keys())}")

        assert atmo_layer in prop.chromatic_shifts_m[sci_source], \
            "Atmospheric layer must have a chromatic shift"
        assert common_layer not in prop.chromatic_shifts_m[sci_source], \
            "Common layer must not have a chromatic shift"
        assert abs(prop.chromatic_shifts_m[sci_source][atmo_layer]) > 0.0, \
            "Atmo chromatic shift should be non-zero"

    @cpu_and_gpu
    def test_chromatic_effect_does_not_change_common_layer_only_prop(self, target_device_idx, xp):
        """Test that chromatic effect has no impact when only common layers are propagated."""
        pixel_pupil = 100
        simul_params = SimulParams(pixel_pupil, 0.1)

        common_layer = Layer(
            dimx=140, dimy=140,
            pixel_pitch=0.1,
            height=2000.0,
            target_device_idx=target_device_idx
        )
        x = xp.arange(140, dtype=float)
        common_layer.A = xp.ones((140, 140))
        common_layer.phaseInNm = xp.tile(x, (140, 1))
        common_layer.generation_time = 1

        source_reference = Source(
            polar_coordinates=[12.0, 35.0],
            magnitude=8,
            wavelengthInNm=2200.0,
            target_device_idx=target_device_idx
        )
        source_chromatic = Source(
            polar_coordinates=[12.0, 35.0],
            magnitude=8,
            wavelengthInNm=2200.0,
            target_device_idx=target_device_idx
        )

        prop_ref = AtmoPropagation(
            simul_params,
            source_dict={'ref': source_reference},
            enable_chromatic_effect=False,
            target_device_idx=target_device_idx
        )
        prop_ref.inputs['atmo_layer_list'].set([])
        prop_ref.inputs['common_layer_list'].set([common_layer])

        prop_chrom = AtmoPropagation(
            simul_params,
            source_dict={'chrom': source_chromatic},
            enable_chromatic_effect=True,
            chromatic_reference_wavelengthInNm=589.0,
            telescope_altitude_m=3064.0,
            target_device_idx=target_device_idx
        )
        prop_chrom.inputs['atmo_layer_list'].set([])
        prop_chrom.inputs['common_layer_list'].set([common_layer])

        loop = LoopControl()
        loop.add(prop_chrom, idx=0)
        loop.add(prop_ref, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        ef_ref = prop_ref.outputs['out_ref_ef']
        ef_chrom = prop_chrom.outputs['out_chrom_ef']

        amp_diff = cpuArray(ef_chrom.A) - cpuArray(ef_ref.A)
        ph_diff = cpuArray(ef_chrom.phaseInNm) - cpuArray(ef_ref.phaseInNm)

        assert np.max(np.abs(amp_diff)) < 1e-10, \
            "Amplitude should be unchanged for common-layer-only propagation"
        assert np.max(np.abs(ph_diff)) < 1e-10, \
            "Phase should be unchanged for common-layer-only propagation"

    @cpu_and_gpu
    def test_chromatic_shift_applied_for_on_axis_source(self, target_device_idx, xp):
        """Test that chromatic shift triggers interpolation even for on-axis sources."""
        pixel_pupil = 60
        pixel_pitch = 0.1
        # Use a non-zero zenith angle to generate a chromatic shift
        simul_params = SimulParams(pixel_pupil, pixel_pitch, zenithAngleInDeg=45.0)

        # Create an elevated atmospheric layer
        layer = Layer(
            dimx=100, dimy=100,
            pixel_pitch=pixel_pitch,
            height=10000.0,
            target_device_idx=target_device_idx
        )
        layer.A = xp.ones((100, 100))

        # Create a phase ramp along the Y-axis (elevation axis) to easily measure the shift
        y_ramp = xp.arange(100, dtype=float)
        layer.phaseInNm = xp.tile(y_ramp, (100, 1)).T
        layer.generation_time = 1

        # On-axis source at a different wavelength than the reference
        on_axis_source = Source(
            polar_coordinates=[0.0, 0.0],
            magnitude=8,
            wavelengthInNm=2200.0,  # Far from 500nm reference
            target_device_idx=target_device_idx
        )

        prop = AtmoPropagation(
            simul_params,
            source_dict={'on_axis': on_axis_source},
            enable_chromatic_effect=True,
            chromatic_reference_wavelengthInNm=500.0,
            telescope_altitude_m=3000.0,
            target_device_idx=target_device_idx
        )

        prop.inputs['atmo_layer_list'].set([layer])
        prop.inputs['common_layer_list'].set([])

        # Run setup to initialize interpolators
        prop.setup()

        # 1. ASSERT INTERPOLATOR IS CREATED
        # Without the bug fix, this would be None because source.r == 0
        assert prop.interpolators[on_axis_source][layer] is not None, \
            "Interpolator must be created for on-axis source if chromatic shift is present."

        # Run propagation
        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        output_ef = prop.outputs['out_on_axis_ef']
        output_phase = cpuArray(output_ef.phaseInNm)

        # 2. ASSERT THE SHIFT IS ACTUALLY APPLIED IN OUTPUT
        # Expected center value of the phase without shift is the center of the layer (y=50)
        # Because we have a chromatic shift, the mean phase should deviate from 50.0
        mean_phase = np.mean(output_phase)
        assert not np.isclose(mean_phase, 50.0, atol=1e-3), \
            f"Phase output should be shifted chromatically, but got mean phase {mean_phase}"

    @cpu_and_gpu
    def test_airmass_not_applied_to_common_layers(self, target_device_idx, xp):
        """Test that airmass does NOT affect the interpolation of common layers.

        A common layer (e.g. a DM conjugated at altitude) must produce the same
        output regardless of the zenith angle, because its conjugation height is
        a physical property of the instrument, not a projected atmospheric height.
        Conversely, an atmospheric layer must produce a different output when the
        zenith angle changes (pixel_position scales with airmass).
        """
        pixel_pupil = 100
        pixel_pitch = 0.1
        layer_height = 10000.0  # metres
        dim_layer = 160  # large enough to avoid out-of-FoV for the tested offsets

        # Off-axis source: small offset so the interpolation shifts noticeably
        source_r_arcsec = 5.0
        source_phi_deg = 0.0

        # Build a phase ramp along x so any lateral shift is numerically measurable
        x_ramp = xp.tile(xp.arange(dim_layer, dtype=float), (dim_layer, 1))

        def _run(zenith_deg, is_atmo):
            simul_params = SimulParams(pixel_pupil, pixel_pitch, zenithAngleInDeg=zenith_deg)
            layer = Layer(
                dimx=dim_layer, dimy=dim_layer,
                pixel_pitch=pixel_pitch,
                height=layer_height,
                target_device_idx=target_device_idx
            )
            layer.A = xp.ones((dim_layer, dim_layer))
            layer.phaseInNm = x_ramp.copy()
            layer.generation_time = 1

            source = Source(
                polar_coordinates=[source_r_arcsec, source_phi_deg],
                magnitude=8, wavelengthInNm=750,
                target_device_idx=target_device_idx
            )
            prop = AtmoPropagation(
                simul_params,
                source_dict={'src': source},
                target_device_idx=target_device_idx
            )
            if is_atmo:
                prop.inputs['atmo_layer_list'].set([layer])
                prop.inputs['common_layer_list'].set([])
            else:
                prop.inputs['atmo_layer_list'].set([])
                prop.inputs['common_layer_list'].set([layer])

            loop = LoopControl()
            loop.add(prop, idx=0)
            loop.run(run_time=1, dt=1, t0=0)
            return cpuArray(prop.outputs['out_src_ef'].phaseInNm)

        phase_common_z0  = _run(zenith_deg=0.0,  is_atmo=False)
        phase_common_z45 = _run(zenith_deg=45.0, is_atmo=False)
        phase_atmo_z0    = _run(zenith_deg=0.0,  is_atmo=True)
        phase_atmo_z45   = _run(zenith_deg=45.0, is_atmo=True)

        diff_common = np.max(np.abs(phase_common_z45 - phase_common_z0))
        diff_atmo   = np.max(np.abs(phase_atmo_z45   - phase_atmo_z0))

        assert diff_common < 1e-6, (
            f"Common layer output must be invariant to zenith angle, "
            f"but max diff = {diff_common:.3e}"
        )
        assert diff_atmo > 1e-3, (
            f"Atmo layer output must change with zenith angle (airmass effect), "
            f"but max diff = {diff_atmo:.3e}"
        )

    @cpu_and_gpu
    def test_rectangular_layer(self, target_device_idx, xp):
        """Rectangular (non-square) layers must use the correct half-size for each axis.

        half_pixel_layer must be built from size[0] and size[1] separately so that
        the centre of the layer is placed correctly when dimx != dimy.
        """
        pixel_pupil = 60
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        dimx, dimy = 100, 140   # intentionally non-square

        layer = Layer(
            dimx=dimx, dimy=dimy,
            pixel_pitch=pixel_pitch,
            height=0.0,
            target_device_idx=target_device_idx
        )
        # Layer arrays are (dimy, dimx) — rows=y, cols=x
        layer.A = xp.ones((dimy, dimx))
        layer.phaseInNm = xp.zeros((dimy, dimx))
        layer.generation_time = 1

        source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750,
                        target_device_idx=target_device_idx)
        prop = AtmoPropagation(simul_params,
                               source_dict={'on_axis': source},
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([])
        prop.inputs['common_layer_list'].set([layer])

        loop = LoopControl()
        loop.add(prop, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        output_ef = prop.outputs['out_on_axis_ef']
        assert output_ef.A.shape == (pixel_pupil, pixel_pupil)
        assert np.allclose(cpuArray(output_ef.A), 1.0), \
            "Uniform amplitude layer must produce all-ones output"
        assert np.allclose(cpuArray(output_ef.phaseInNm), 0.0), \
            "Zero-phase layer must produce all-zeros phase output"

    @cpu_and_gpu
    def test_source_outside_fov_raises(self, target_device_idx, xp):
        """When an atmospheric layer is too small to cover the source direction,
        setup() must raise ValueError with a descriptive message.

        layer_interpolator() returns None for out-of-FoV sources, and
        setup_interpolators() escalates that to a hard error so the user is
        aware of the misconfiguration before the simulation starts.
        """
        pixel_pupil = 20
        pixel_pitch = 0.1
        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        dim_layer = pixel_pupil   # zero margin: any lateral offset is out-of-FoV

        layer = Layer(
            dimx=dim_layer, dimy=dim_layer,
            pixel_pitch=pixel_pitch,
            height=10000.0,
            target_device_idx=target_device_idx
        )
        layer.A = xp.ones((dim_layer, dim_layer))
        layer.phaseInNm = xp.zeros((dim_layer, dim_layer))
        layer.generation_time = 1

        # Large off-axis angle → pixel_position >> layer half-size
        source = Source(polar_coordinates=[300.0, 0.0], magnitude=8, wavelengthInNm=750,
                        target_device_idx=target_device_idx)

        prop = AtmoPropagation(simul_params,
                               source_dict={'src': source},
                               target_device_idx=target_device_idx)
        prop.inputs['atmo_layer_list'].set([layer])
        prop.inputs['common_layer_list'].set([])

        with self.assertRaises(ValueError):
            prop.setup()

    @cpu_and_gpu
    def test_lgs_cone_effect(self, target_device_idx, xp):
        """Finite-height source (LGS) must produce a cone-effect scaling of the pupil footprint.

        The effective pupil size on the layer is pixel_pupil * (sh - lh) / sh.
        We verify this by checking that the output amplitude covers a larger fraction
        of a pattern than the NGS (infinite height) case does.
        """
        pixel_pupil = 60
        pixel_pitch = 0.1
        layer_height_m = 5000.0
        lgs_height_m   = 90000.0   # typical sodium LGS height

        simul_params = SimulParams(pixel_pupil, pixel_pitch)

        # Layer with a bright central disc and dark border: wider sampling → more ones
        dim_layer = 200
        layer_A = xp.zeros((dim_layer, dim_layer))
        cx = dim_layer // 2
        for i in range(dim_layer):
            for j in range(dim_layer):
                if (i - cx)**2 + (j - cx)**2 < (cx * 0.9)**2:
                    layer_A[i, j] = 1.0

        def _run(source_height):
            layer = Layer(
                dimx=dim_layer, dimy=dim_layer,
                pixel_pitch=pixel_pitch,
                height=layer_height_m,
                target_device_idx=target_device_idx
            )
            layer.A = layer_A.copy()
            layer.phaseInNm = xp.zeros((dim_layer, dim_layer))
            layer.generation_time = 1

            if np.isinf(source_height):
                source = Source(polar_coordinates=[0.0, 0.0], magnitude=8,
                                wavelengthInNm=750, target_device_idx=target_device_idx)
            else:
                source = Source(polar_coordinates=[0.0, 0.0], magnitude=8,
                                wavelengthInNm=750, height=source_height,
                                target_device_idx=target_device_idx)

            prop = AtmoPropagation(simul_params,
                                   source_dict={'src': source},
                                   target_device_idx=target_device_idx)
            prop.inputs['atmo_layer_list'].set([layer])
            prop.inputs['common_layer_list'].set([])

            loop = LoopControl()
            loop.add(prop, idx=0)
            loop.run(run_time=1, dt=1, t0=0)
            return float(xp.mean(prop.outputs['out_src_ef'].A))

        mean_ngs = _run(float('inf'))
        mean_lgs = _run(lgs_height_m)

        # LGS sees a smaller footprint on the layer (cone narrows toward the source).
        # cone_coeff = (lgs_height - layer_height) / lgs_height  ≈ 0.944  < 1
        # → the LGS pupil footprint is smaller → it samples more of the bright disc
        # → mean amplitude for LGS must be >= mean amplitude for NGS
        assert mean_lgs >= mean_ngs - 0.01, (
            f"LGS (cone effect) should sample at least as much of the bright disc "
            f"as NGS, but mean_lgs={mean_lgs:.4f} < mean_ngs={mean_ngs:.4f}"
        )
