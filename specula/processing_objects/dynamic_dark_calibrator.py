import os
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams
from specula.lib import utils


class DynamicDarkCalibrator(BaseProcessingObj):
    """
    Dark calibrator processing object. Calibrator for pixel dark frames.
    """
    def __init__(self,
                 data_dir: str,      # Set by main Simul object
                 nframes: int,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if nframes <= 0:
            raise ValueError(f'Number of frames is {nframes} and must be greater than zero')

        self.nframes = nframes
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.integrated_pixels = None
        self.counter = 0

        # Inputs
        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_trigger'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_nframes'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_load'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_save'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_reset'] = InputValue(type=BaseValue, optional=True)

        # Outputs
        self.darkframe = Pixels(
            dimx=0, dimy=0,  # Will be set after first trigger
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        self.subtracted_pixels = Pixels(
            dimx=0, dimy=0,  # Will be set after first trigger
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        self.outputs['out_darkframe'] = self.darkframe
        self.outputs['out_subtracted_pixels'] = self.subtracted_pixels

    def setup(self):
        """Resize output darkframe to match input pixel dimensions and properties"""
        super().setup()

        in_pixels = self.local_inputs['in_pixels']

        dimy, dimx = in_pixels.size
        self.darkframe.resize(dimx,
                              dimy,
                              in_pixels.bpp,
                              in_pixels.signed)
        self.subtracted_pixels.resize(dimx,
                                      dimy,
                                      in_pixels.bpp,
                                      in_pixels.signed)
        self.integrated_pixels = self.darkframe.pixels * 0

    def trigger_code(self):
        """Main calibration function"""
        
        value = self.local_inputs['in_pixels'].pixels

        self.subtracted_pixels.pixels = value - self.darkframe.pixels
        self.subtracted_pixels.generation_time = self.current_time

        if self.counter == 0:
            return

        self.integrated_pixels += value
        self.counter -= 1

        if self.counter == 0:
            self.darkframe.pixels[:] = (self.integrated_pixels / self.nframes).astype(self.darkframe.pixels.dtype)
            self.darkframe.generation_time = self.current_time
            self.integrated_pixels *= 0

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Check if new trigger or nframes value is received at this time step
        input_trigger = self.local_inputs['in_trigger']
        if input_trigger is not None and input_trigger.generation_time == self.current_time:
            self.counter = self.nframes

        input_reset = self.local_inputs['in_reset']
        if input_reset is not None and input_reset.generation_time == self.current_time:
            self.darkframe.pixels *= 0

        # Interactive inputs are protected frome exceptions
        try:
            input_nframes = self.local_inputs['in_nframes']
            if input_nframes is not None and input_nframes.generation_time == self.current_time:
                nframes = int(input_nframes.value)
                if nframes <= 0:
                    raise ValueError(f'Number of frames is {nframes} and must be greater than zero')
                self.nframes = nframes

            input_load = self.local_inputs['in_load']
            if input_load is not None and input_load.generation_time == self.current_time:
                filename = str(input_load.value)
                if not filename.endswith('.fits'):
                    filename += '.fits'
                fullpath = os.path.join(self.data_dir, filename)
                self.darkframe.restore(fullpath)
                self.darkframe.generation_time = self.current_time
        except Exception as e:
            print(f'Exception: {e.__name__}: {e}')

    def post_trigger(self):
        super().post_trigger()

        input_save = self.local_inputs['in_save']
        if input_save is not None and input_save.generation_time == self.current_time:
            self.save(input_save.value)

    def save(self, filename=None):
        """Save dark frame data to disk as a FITS file"""

        if filename is None:
            filename = utils.make_tn()

        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self.data_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.darkframe.save(file_path, overwrite=self.overwrite)

        if self.verbose:
            print(f'Saved dark frame data: {file_path}')


