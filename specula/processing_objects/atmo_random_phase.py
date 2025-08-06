import array
import numpy as np

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.electric_field import ElectricField
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

class AtmoRandomPhase(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 L0: float=1.0,
                 data_dir: str="", 
                 source_dict: dict={},
                 wavelengthInNm: float=500.0,                 
                 pixel_phasescreens=None,
                 seed: int=1,
                 target_device_idx=None,
                 precision=None,
                 verbose=None):


        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
       
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.zenithAngleInDeg = self.simul_params.zenithAngleInDeg

        self.source_dict = source_dict
        self.last_position = 0
        self.seeing = 1
        self.airmass = 1
        self.wavelengthInNm = wavelengthInNm
        self.seed = seed
        
        self.inputs['seeing'] = InputValue(type=BaseValue)
        
        if self.zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(self.zenithAngleInDeg))
            print(f'AtmoRandomPhase: zenith angle is defined as: {self.zenithAngleInDeg} deg')
            print(f'AtmoRandomPhase: airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        # Compute layers dimension in pixels
        self.pixel_layer_size = self.pixel_pupil

        self.L0 = L0        
        self.data_dir = data_dir
        self.seeing = None

        if pixel_phasescreens is None:
            self.pixel_square_phasescreens = 8192
        else:
            self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < self.pixel_layer_size:
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')
        
        self.verbose = verbose if verbose is not None else False
        
        # Initialize layer list with correct heights
        self.layer_list = []
        layer = Layer(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, 0, precision=self.precision, target_device_idx=self.target_device_idx)
        self.layer_list.append(layer)
        
        for name, source in source_dict.items():
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, target_device_idx=self.target_device_idx)
            ef.S0 = source.phot_density()
            self.outputs['out_'+name+'_ef'] = ef

        if self.seed < 1:
            raise ValueError('Seed must be >1')

        self.initScreens()

        self.inputs['pupilstop'] = InputValue(type=Pupilstop)
    

    def initScreens(self):
        # Seed
        if type(self.seed) is not np.ndarray:
            self.seed = np.array([self.seed])
        # Square phasescreens
        square_phasescreens = phasescreens_manager(np.array([self.L0]), self.pixel_square_phasescreens,
                                                    self.pixel_pitch, self.data_dir,
                                                    seed=self.seed, precision=self.precision,
                                                    verbose=self.verbose, xp=self.xp)
        # number of slices to be cut from the 2D array
        num_slices = (self.pixel_square_phasescreens // self.pixel_pupil)

        # it cuts the array to have dimensions multiple of pixel_pupil
        input_array = square_phasescreens[0][0:num_slices*self.pixel_pupil,0:num_slices*self.pixel_pupil]

        # it makes a 3D array stacking neighbouring squares of the 2D array
        temp_screen = input_array.reshape(num_slices, self.pixel_pupil,num_slices, self.pixel_pupil).swapaxes(1, 2).reshape(-1, self.pixel_pupil, self.pixel_pupil)

        # phase in rad
        temp_screen *= self.wavelengthInNm / (2 * np.pi)

        temp_screen = self.to_xp(temp_screen, dtype=self.dtype)
        
        self.phasescreens = temp_screen

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.pupilstop = self.local_inputs['pupilstop']
    
    def trigger_code(self):
        r0 = 0.9759 * 0.5 / (self.local_inputs['seeing'].value * 4.848) * self.airmass**(-3./5.) # if seeing > 0 else 0.0
        r0wavelength = r0 * (self.wavelengthInNm / 500.0)**(6./5.)
        scale_coeff = (self.pixel_pitch / r0wavelength)**(5./6.) # if seeing > 0 else 0.0

        new_position = self.last_position
        if new_position+1 > self.phasescreens.shape[0]:
            self.seed += 1
            self.initScreens()
            new_position = 0

        for name, source in self.source_dict.items():
            self.outputs['out_'+name+'_ef'].phaseInNm = self.phasescreens[new_position,:,:] * scale_coeff
            self.outputs['out_'+name+'_ef'].A = self.pupilstop.A
            self.outputs['out_'+name+'_ef'].generation_time = self.current_time

        # Update position output
        self.last_position = new_position + 1


                

