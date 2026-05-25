import numpy as np

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.data_objects.electric_field import ElectricField
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

class AtmoRandomPhase(BaseProcessingObj):
    """
    Atmospheric random phase screen generator processing object.
    Atmospheric phase screen generator producing random (uncorrelated) phase screens.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 L0: float=1.0,
                 data_dir: str="",
                 source_dict: dict=None,
                 wavelengthInNm: float=500.0,
                 pixel_phasescreens: int=8192,
                 seed: int=1,
                 update_interval: int=1,
                 layer_height: float=0.0,
                 target_device_idx=None,
                 precision=None):
        """
        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing pupil size, pixel pitch, zenith angle, etc.
        L0 : float [m], optional
            Outer scale of turbulence in meters, by default 1.0
        data_dir : str
            Directory path for storing/loading phase screen data (automatically set by simul.py).
        source_dict : dict [1], optional
            Dictionary of sources for the atmospheric phase screens.
            If omitted or empty, the object exposes a single pair of outputs named
            out_ef and out_layer.
        wavelengthInNm : float [nm], optional
            Wavelength in nanometers for scaling the phase screens, by default 500.0 nm
        pixel_phasescreens : int [1], optional
            Size of the square phase screens in pixels. Defaults to 8192.
        seed : int [1], optional
            Seed for random number generation, by default 1.
        update_interval : int [1], optional
            Number of triggers between phase screen updates, by default 1.
        layer_height : float [m], optional
            Height in meters assigned to the output layers, by default 0.0.
        target_device_idx : int [1], optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int [1], optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.pixel_pupil = simul_params.pixel_pupil
        self.pixel_pitch = simul_params.pixel_pitch
        zenithAngleInDeg = simul_params.zenithAngleInDeg

        source_dict = source_dict or {}
        self.new_position = 0
        self.last_position = 0
        self.update_interval = update_interval
        self.step_counter = 0
        self.wavelengthInNm = wavelengthInNm
        self.scale_coeff = 1.0
        self.seed = seed
        self.layer_outputs = {}

        self.pupilstop = None

        self.inputs['seeing'] = InputValue(type=BaseValue)

        if zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg))
            self.logger.info(f'AtmoRandomPhase: zenith angle is defined as: {zenithAngleInDeg} deg')
            self.logger.info(f'AtmoRandomPhase: airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        self.L0 = L0
        self.data_dir = data_dir
        self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < self.pixel_pupil:
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')

        output_specs = list(source_dict.items()) if source_dict else [(None, None)]

        for name, source in output_specs:
            layer_output_name = 'out_layer' if name is None else 'out_'+name+'_layer'
            ef_output_name = 'out_ef' if name is None else 'out_'+name+'_ef'

            layer = Layer(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, layer_height,
                          precision=self.precision, target_device_idx=self.target_device_idx)
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                               target_device_idx=self.target_device_idx)
            # The electric field output shares the same array as the layer output
            ef.field = layer.field
            if source is not None:
                ef.S0 = source.phot_density()

            self.layer_outputs[layer_output_name] = layer
            self.outputs[layer_output_name] = layer
            self.outputs[ef_output_name] = ef

        if self.seed < 1:
            raise ValueError('Seed must be >1')

        self.initScreens()

        self.inputs['pupilstop'] = InputValue(type=Pupilstop)

    @classmethod
    def input_names(cls):
        return {'seeing': InputDesc(BaseValue, 'Atmospheric seeing value'),
                'pupilstop': InputDesc(Pupilstop, 'Pupil stop mask defining the valid pupil area')}

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

    def initScreens(self):
        # Seed
        if type(self.seed) is not np.ndarray:
            self.seed = np.array([self.seed])
        # Square phasescreens
        square_phasescreens = phasescreens_manager(np.array([self.L0]),
                                                   self.pixel_square_phasescreens,
                                                   self.pixel_pitch, self.data_dir,
                                                   seed=self.seed, precision=self.precision,
                                                   xp=self.xp)
        # number of slices to be cut from the 2D array
        num_slices = self.pixel_square_phasescreens // self.pixel_pupil

        # it cuts the array to have dimensions multiple of pixel_pupil
        input_array = square_phasescreens[0][0:num_slices*self.pixel_pupil,
                                             0:num_slices*self.pixel_pupil]

        # it makes a 3D array stacking neighbouring squares of the 2D array
        temp_screen = input_array.reshape(
            num_slices, self.pixel_pupil, num_slices, self.pixel_pupil
        ).swapaxes(1, 2).reshape(-1, self.pixel_pupil, self.pixel_pupil)

        # phase in rad
        temp_screen *= self.wavelengthInNm / (2 * np.pi)

        temp_screen = self.to_xp(temp_screen, dtype=self.dtype)

        self.phasescreens = temp_screen

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.pupilstop = self.local_inputs['pupilstop']

        r0 = 0.9759 * 0.5 / (self.local_inputs['seeing'].value * 4.848) \
             * self.airmass**(-3./5.) # if seeing > 0 else 0.0
        r0wavelength = r0 * (self.wavelengthInNm / 500.0)**(6./5.)
        self.scale_coeff = (self.pixel_pitch / r0wavelength)**(5./6.) # if seeing > 0 else 0.0

        self.new_position = self.last_position

        if self.new_position >= self.phasescreens.shape[0]:
            self.seed += 1
            self.initScreens()
            self.new_position = 0

    def trigger_code(self):
        current_phase = self.phasescreens[self.new_position,:,:] * self.scale_coeff

        for output_name, layer in self.layer_outputs.items():
            layer.phaseInNm[:] = current_phase
            layer.A[:] = self.pupilstop.A
            layer.generation_time = self.current_time

            # Update the corresponding electric field output generation time
            # Note: the electric field output shares the same array (ef.field)
            #       as the layer output (layer.field)
            ef_output_name = output_name.replace('_layer', '_ef')
            self.outputs[ef_output_name].generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()

        # increment step counter and check if update is needed
        self.step_counter += 1
        if self.step_counter % self.update_interval == 0:
            self.last_position = self.new_position+1
