import numpy as np
from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.electric_field import ElectricField
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.extrapolation_2d import EFInterpolator

class PhaseScreenCube(BaseProcessingObj):
    """
    User-defined phase screen cube data object.
    Reads a phase screen cube from a FITS file and applies it on the specified line of sight.
    The cube's temporal sampling does not need to match the simulation's sampling.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 file_name: str,
                 time_step: float,
                 pixel_scale: float,
                 source_dict: dict,
                 verbose=None,
                 target_device_idx=None):
        """
        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing pupil size, pixel pitch, zenith angle, etc.
        file_name : str
            Full path to a FITS file containing the phase screen cube. The cube should 
            have the temporal evolution on the third dimension. The phase screens should be in nm.
        time_step : float
            Time resolution of the phase screen cube in seconds.
        pixel_scale : float
            Phase screens' pixel size in m.
        source_dict : dict
            Dictionary of the source corresponding to the line of sight of the phase screen.
        verbose : bool, optional
            If True, enables verbose output during phase screen generation.
            Default is None (no verbose output).
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx)

        self.simul_params = simul_params

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

        self.source_dict = source_dict
        self.step_counter = 0
        
        self.pupilstop = None

        self.file_name = file_name
        self.time_step = time_step
        self.pixel_scale = pixel_scale

        self.verbose = verbose if verbose is not None else False

        # Initialize layer list
        self.layer_list = []
        layer = Layer(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, 0,
                      target_device_idx=self.target_device_idx)
        self.layer_list.append(layer)

        for name, source in source_dict.items():
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                               target_device_idx=self.target_device_idx)
            ef.S0 = source.phot_density()
            self.outputs['out_'+name+'_ef'] = ef

        self.initScreens()

        self.inputs['pupilstop'] = InputValue(type=Pupilstop)

    def initScreens(self):
        with fits.open(self.file_name) as hdul:
            temp_screen = hdul[0].data.T.astype(self.dtype)

        self.phasescreens = temp_screen
        
        dim = temp_screen.shape
        self.time_vector = np.arange(dim[2])*self.time_step

        self.scaling_fact = dim[0]/self.pixel_pupil*self.pixel_scale/self.pixel_pitch

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.pupilstop = self.local_inputs['pupilstop']

        if self.t_to_seconds(t) > np.max(self.time_vector):
            raise ValueError('Error: the simulation is too long with respect to the input phase screen cube!')
        
        dt = self.time_vector-self.t_to_seconds(t)
        idx_first_positive = np.searchsorted(dt, 0, side='right')
        if idx_first_positive >= len(dt):
            idx_first_positive = len(dt)-1
        idx_last_non_positive = idx_first_positive - 1

        self.cur_screen = 1./self.time_step*(dt[idx_first_positive]*self.phasescreens[:,:,idx_last_non_positive] + 
                                            np.abs(dt[idx_last_non_positive])*self.phasescreens[:,:,idx_first_positive])

        in_ef = ElectricField(self.cur_screen.shape[0], self.cur_screen.shape[1], self.pixel_scale,
                               target_device_idx=self.target_device_idx)
        
        in_ef.phaseInNm = self.cur_screen

        self.ef_interpolator = EFInterpolator(
            in_ef,
            (self.pixel_pupil,self.pixel_pupil),
            magnification = self.scaling_fact,
            target_device_idx=self.target_device_idx,
            use_out_ef_cache=False, # we cannot reuse the cache here because the interpolated array
                                    # is computed in prepare_trigger, but is used in trigger_code
        )

        self.ef_interpolator.interpolate()


    def trigger_code(self):
        for name, source in self.source_dict.items():
            self.outputs['out_'+name+'_ef'].phaseInNm = self.ef_interpolator.interpolated_ef().phaseInNm
            self.outputs['out_'+name+'_ef'].A = self.pupilstop.A
            self.outputs['out_'+name+'_ef'].generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()
