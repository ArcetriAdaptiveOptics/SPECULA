import numpy as np

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.spatio_temp_array import SpatioTempArray
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.extrapolation_2d import EFInterpolator

class PhaseScreenCube(BaseProcessingObj):
    """
    User-defined phase screen cube data object.
    Applies a spatio-temporal phase screen cube on the specified line of sight.
    The cube's temporal sampling does not need to match the simulation's sampling.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 cube: SpatioTempArray,
                 pixel_scale: float,
                 source_dict: dict=None,
                 layer_height: float=0.0,
                 scale_factor: float=1.0,
                 verbose=None,
                 target_device_idx=None):
        """
        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing pupil size, pixel pitch, zenith angle, etc.
        cube : SpatioTempArray
            Spatio-temporal array containing the phase screen cube.
            Internally data are accessed as time-first: shape (time, x, y).
            The phase screens should be in nm. The time_vector must be provided in seconds.
        pixel_scale : float
            Phase screens' pixel size in m.
        source_dict : dict, optional
            Dictionary of the source corresponding to the line of sight of the phase screen.
            If omitted or empty, the object exposes a single pair of outputs named
            out_ef and out_layer.
        layer_height : float, optional
            Height in meters assigned to the output layer, by default 0.0.
        scale_factor : float, optional
            Scaling factor applied to the phase screens, by default 1.0. This can be used 
            to adjust the amplitude of the phase screens if needed.
        verbose : bool, optional
            If True, enables verbose output during phase screen generation.
            Default is None (no verbose output).
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx)

        self.simul_params = simul_params
        self.cube = cube

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.pixel_scale = pixel_scale
        self.scale_factor = scale_factor

        self.source_dict = source_dict or {}
        self.step_counter = 0
        self.layer_height = layer_height
        self.layer_outputs = {}
        self.ef_outputs = {}

        self.pupilstop = None

        self.verbose = verbose if verbose is not None else False

        output_specs = list(self.source_dict.items()) if self.source_dict else [(None, None)]

        for name, source in output_specs:
            layer_output_name = 'out_layer' if name is None else 'out_'+name+'_layer'
            ef_output_name = 'out_ef' if name is None else 'out_'+name+'_ef'

            layer = Layer(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, self.layer_height,
                          target_device_idx=self.target_device_idx)
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                               target_device_idx=self.target_device_idx)
            # The electric field output shares the same array as the layer output
            ef.field = layer.field
            if source is not None:
                ef.S0 = source.phot_density()

            self.layer_outputs[layer_output_name] = layer
            self.ef_outputs[ef_output_name] = ef
            self.outputs[layer_output_name] = layer
            self.outputs[ef_output_name] = ef

        self.initScreens()

        self.inputs['pupilstop'] = InputValue(type=Pupilstop)

    def initScreens(self):
        """
        Initialize phase screens from the cube data object.
        Computes the scaling factor to map the cube spatial dimensions to the pupil grid.
        """
        self.phasescreens = self.to_xp(self.cube.array, dtype=self.dtype)
        self.time_vector = self.to_xp(self.cube.time_vector)

        dim = self.phasescreens.shape
        self.bin_fact = dim[1]/self.pixel_pupil*self.pixel_scale/self.pixel_pitch

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

        # Linear interpolation between two time steps
        time_step = self.time_vector[idx_first_positive] - self.time_vector[idx_last_non_positive]
        self.cur_screen = self.scale_factor/time_step*(dt[idx_first_positive]*self.phasescreens[idx_last_non_positive, :, :] + 
                        np.abs(dt[idx_last_non_positive])*self.phasescreens[idx_first_positive, :, :])

        in_ef = ElectricField(self.cur_screen.shape[0], self.cur_screen.shape[1], self.pixel_scale,
                               target_device_idx=self.target_device_idx)

        in_ef.phaseInNm = self.cur_screen

        self.ef_interpolator = EFInterpolator(
            in_ef,
            (self.pixel_pupil,self.pixel_pupil),
            magnification = self.bin_fact,
            target_device_idx=self.target_device_idx,
            use_out_ef_cache=False, # we cannot reuse the cache here because the interpolated array
                                    # is computed in prepare_trigger, but is used in trigger_code
        )

        self.ef_interpolator.interpolate()


    def trigger_code(self):
        current_phase = self.ef_interpolator.interpolated_ef().phaseInNm
        for output_name, layer in self.layer_outputs.items():
            layer.phaseInNm[:] = current_phase
            layer.A[:] = self.pupilstop.A
            layer.generation_time = self.current_time

            # Update the corresponding electric field output generation time
            # Note: the electric field output shares the same array (ef.field)
            #       as the layer output (layer.field)
            ef_output_name = output_name.replace('_layer', '_ef')
            self.ef_outputs[ef_output_name].generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()
