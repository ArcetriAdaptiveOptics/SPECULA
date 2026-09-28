import numpy as np

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.layer import Layer
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
                 target_device_idx=None,
                 precision=None):
        """
        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing pupil size, pixel pitch, zenith angle, etc.
        cube : SpatioTempArray
            Spatio-temporal array containing the phase screen cube.
            Internally data are accessed as time-first: shape (time, x, y).
            The phase screens should be in nm. The time_vector must be provided in seconds.
        pixel_scale : float [m]
            Phase screens' pixel size in m.
        source_dict : dict [1], optional
            Dictionary of the source corresponding to the line of sight of the phase screen.
            If omitted or empty, the object exposes a single pair of outputs named
            out_ef and out_layer.
        layer_height : float [m], optional
            Height in meters assigned to the output layer, by default 0.0.
        scale_factor : float [1], optional
            Scaling factor applied to the phase screens, by default 1.0. This can be used 
            to adjust the amplitude of the phase screens if needed.
        target_device_idx : int [1], optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int [1], optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.cube = cube

        self.pixel_pupil = simul_params.pixel_pupil
        self.pixel_pitch = simul_params.pixel_pitch
        self.pixel_scale = pixel_scale
        self.scale_factor = scale_factor
        self.layer_outputs = {}

        source_dict = source_dict or {}

        output_specs = list(source_dict.items()) if source_dict else [(None, None)]

        for name, source in output_specs:
            layer_output_name = 'out_layer' if name is None else 'out_'+name+'_layer'
            ef_output_name = 'out_ef' if name is None else 'out_'+name+'_ef'

            layer = Layer(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, layer_height,
                          target_device_idx=self.target_device_idx, precision=self.precision)
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                               target_device_idx=self.target_device_idx, precision=self.precision)
            # The electric field output shares the same array as the layer output
            ef.field = layer.field
            if source is not None:
                ef.S0 = source.phot_density()

            self.layer_outputs[layer_output_name] = layer
            self.outputs[layer_output_name] = layer
            self.outputs[ef_output_name] = ef

        self.initScreens()

    def initScreens(self):
        """
        Initialize phase screens from the cube data object.
        Computes the scaling factor to map the cube spatial dimensions to the pupil grid.
        """
        self.phasescreens = self.to_xp(self.cube.array, dtype=self.dtype)
        # Host float64: only used to find the interpolation indices and weights
        self.time_vector = np.asarray(cpuArray(self.cube.time_vector), dtype=np.float64)

        dim = self.phasescreens.shape
        self.bin_fact = dim[1]/self.pixel_pupil*self.pixel_scale/self.pixel_pitch

        # Built once: the interpolator keeps a reference to cur_screen and reads it at each
        # interpolate(), and the edge extrapolation data (amplitude only) never change
        self.cur_screen = ElectricField(dim[1], dim[2], self.pixel_scale,
                                        target_device_idx=self.target_device_idx, precision=self.precision)
        self.ef_interpolator = EFInterpolator(
            self.cur_screen,
            (self.pixel_pupil, self.pixel_pupil),
            magnification=self.bin_fact,
            target_device_idx=self.target_device_idx,
            precision=self.precision,
            use_out_ef_cache=False,  # a cached output could be overwritten by another interpolator
        )

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        t_seconds = self.t_to_seconds(t)
        if t_seconds > self.time_vector[-1]:
            raise ValueError('Error: the simulation is too long with respect to the input phase screen cube!')
        if t_seconds < self.time_vector[0]:
            raise ValueError('Error: the simulation starts before the input phase screen cube!')

        dt = self.time_vector - t_seconds
        idx_first_positive = int(np.searchsorted(dt, 0, side='right'))
        if idx_first_positive >= len(dt):
            idx_first_positive = len(dt)-1
        idx_last_non_positive = idx_first_positive - 1

        # Linear interpolation between two time steps, with Python float weights
        time_step = self.time_vector[idx_first_positive] - self.time_vector[idx_last_non_positive]
        w_last = float(self.scale_factor * dt[idx_first_positive] / time_step)
        w_first = float(self.scale_factor * abs(dt[idx_last_non_positive]) / time_step)
        self.cur_screen.phaseInNm[:] = w_last * self.phasescreens[idx_last_non_positive, :, :] + \
                                       w_first * self.phasescreens[idx_first_positive, :, :]

        self.ef_interpolator.interpolate()

    @classmethod
    def output_names(cls):
        return {
            'out_{source_name_}layer': OutputDesc(
                Layer,
                'Output phase-screen layer for named source [source_name]; if source name is None, key is out_layer',
            ),
            'out_{source_name_}ef': OutputDesc(
                ElectricField,
                'Output electric field for named source [source_name]; if source name is None, key is out_ef',
            ),
        }

    def trigger_code(self):
        current_phase = self.ef_interpolator.interpolated_ef().phaseInNm
        for output_name, layer in self.layer_outputs.items():
            layer.phaseInNm[:] = current_phase
            layer.generation_time = self.current_time

            # Update the corresponding electric field output generation time
            # Note: the electric field output shares the same array (ef.field)
            #       as the layer output (layer.field)
            ef_output_name = output_name.replace('_layer', '_ef')
            self.outputs[ef_output_name].generation_time = self.current_time
