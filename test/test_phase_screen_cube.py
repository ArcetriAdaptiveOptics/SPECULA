import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray
from specula import np
from astropy.io import fits

from specula.data_objects.source import Source
from specula.base_time_obj import BaseTimeObj
from specula.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.spatio_temp_array import SpatioTempArray
from specula.processing_objects.phase_screen_cube import PhaseScreenCube

from test.specula_testlib import cpu_and_gpu


class TestPhaseScreenCube(unittest.TestCase):

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    @staticmethod
    def load_cube_from_fits(fits_file, target_device_idx=-1):
        """
        Load phase screen cube from FITS file into a SpatioTempArray.
        
        Parameters
        ----------
        fits_file : str
            Path to FITS file with phase screen cube.
        target_device_idx : int
            Target device index for the array.
            
        Returns
        -------
        SpatioTempArray
            Cube with shape (x, y, time).
        """
        with fits.open(fits_file) as hdul:
            cube_data = hdul[0].data.T.astype(np.float64)

        # cube_data has shape (x, y, time) after transpose
        time_vector = np.arange(cube_data.shape[2]) * 10  # time_step = 10 seconds

        return SpatioTempArray(cube_data, time_vector, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def setUp(self, target_device_idx, xp):
        self.simul_params = SimulParams(pixel_pupil=80, pixel_pitch=0.05, time_step=1)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        source_dict = {'on_axis_source': on_axis_source}

        # Load phase screen cube from FITS into SpatioTempArray
        fits_file = os.path.join(self.data_dir, 'phase_screen_cube_test.fits')
        cube = self.load_cube_from_fits(fits_file, target_device_idx=target_device_idx)

        self.cube = PhaseScreenCube(self.simul_params,
                                    cube=cube,
                                    pixel_scale=0.1,
                                    source_dict=source_dict,
                                    target_device_idx=target_device_idx)

        pupilstop = Pupilstop(self.simul_params,
                              target_device_idx=target_device_idx)
        self.cube.inputs['pupilstop'].set(pupilstop)

        pupilstop.generation_time = 4e9

    @cpu_and_gpu
    def test_screen_cube(self, target_device_idx, xp):
        '''Test that the phase screen is correctly read and interpolated for the current step'''

        self.cube.check_ready(4e9)
        self.cube.trigger()
        self.cube.post_trigger()

        answer0 = self.cube.cur_screen
        answer1 = self.cube.outputs['out_on_axis_source_ef'].phaseInNm

        assert 'out_on_axis_source_layer' in self.cube.outputs
        assert self.cube.outputs['out_on_axis_source_layer'].field is self.cube.outputs['out_on_axis_source_ef'].field

        #Expected
        screen0 = np.transpose(np.resize(np.linspace(-1,1,100),(100,100)))
        screen1 = np.transpose(screen0)*2
        #screen2 = -3*screen0

        screen0b = np.transpose(np.resize(np.linspace(-1,1,80),(80,80)))
        screen1b = np.transpose(screen0b)*2
        #screen2b = -3*screen0b

        exp_screen0 = 0.4*screen1+0.6*screen0
        exp_screen1 = (0.4*screen1b+0.6*screen0b)*80/100.*0.05/0.1

        np.testing.assert_array_almost_equal(cpuArray(answer0),
                                             cpuArray(exp_screen0))
        np.testing.assert_array_almost_equal(cpuArray(answer1),
                                             cpuArray(exp_screen1),
                                             decimal = 2)

    @cpu_and_gpu
    def test_default_output_names_without_source_dict(self, target_device_idx, xp):
        fits_file = os.path.join(self.data_dir, 'phase_screen_cube_test.fits')
        cube = self.load_cube_from_fits(fits_file, target_device_idx=target_device_idx)

        phase_screen_cube = PhaseScreenCube(self.simul_params,
                                            cube=cube,
                                            pixel_scale=0.1,
                                            source_dict=None,
                                            target_device_idx=target_device_idx)

        assert 'out_ef' in phase_screen_cube.outputs
        assert 'out_layer' in phase_screen_cube.outputs
        assert phase_screen_cube.outputs['out_layer'].field is phase_screen_cube.outputs['out_ef'].field
