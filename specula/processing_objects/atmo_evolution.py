import numpy as np

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.connections import InputValue
from specula import cpuArray, ASEC2RAD
from specula.data_objects.simul_params import SimulParams


# Phasescreens are always defined at 500 nm
ATMO_WAVELENGTH = 500.0


class AtmoEvolution(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 L0: list,           # TODO =[1.0],
                 heights: list,      # TODO =[0.0],
                 Cn2: list,          # TODO =[1.0],
                 data_dir: str,      # TODO ="",
                 fov: float=0.0,
                 pixel_phasescreens: int=8192,
                 seed: int=1,
                 extra_delta_time: float=0,
                 verbose: bool=False,
                 fov_in_m: float=None,
                 pupil_position:list =[0,0],
                 target_device_idx: int=None,
                 precision: int=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self.simul_params = simul_params       

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.zenithAngleInDeg = self.simul_params.zenithAngleInDeg

        self.n_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_phasescreens)
        self.last_t = 0
        self.cycle_screens = True
        self.delta_time = None
        self.extra_delta_time = extra_delta_time
                
        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)
        
        if self.zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(self.zenithAngleInDeg), dtype=self.dtype)
            print(f'Atmo_Evolution: zenith angle is defined as: {self.zenithAngleInDeg} deg')
            print(f'Atmo_Evolution: airmass is: {self.airmass}')   
        else:
            self.airmass = 1.0

        heights = np.array(heights, dtype=self.dtype)

        # TODO old code
        fov_rad = fov * ASEC2RAD
        self.pixel_layer = np.ceil((self.pixel_pupil + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) / self.pixel_pitch + 
                               abs(heights) / self.pixel_pitch * fov_rad) / 2.0) * 2.0

        # TODO new code to be tested
        #  
        #  # Conversion coefficient from arcseconds to radians
        #  sec2rad = 4.848e-6
        #          
        #  alpha_fov = fov / 2.0
        #  
        #  # Max star angle from arcseconds to radians
        #  rad_alpha_fov = alpha_fov * sec2rad
        #   
        #  # Compute layers dimension in pixels
        #  self.pixel_layer = np.ceil((pixel_pupil + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) / pixel_pitch + 
        #                         2.0 * abs(heights) / pixel_pitch * rad_alpha_fov) / 2.0) * 2.0

        if fov_in_m is not None:
            self.pixel_layer = np.full_like(heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2)
        
        self.L0 = L0
        self.heights = heights
        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.pixel_pupil = self.pixel_pupil
        self.data_dir = data_dir

        # TODO old code
        self.pixel_square_phasescreens = pixel_phasescreens

        # TODO new code to be tested
        # if pixel_phasescreens is None:
        #     self.pixel_square_phasescreens = 8192
        # else:
        #     self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < max(self.pixel_layer):
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')

        self.verbose = verbose

        # Initialize layer list with correct heights
        self.layer_list = []
        for i in range(self.n_phasescreens):
            layer = Layer(self.pixel_layer[i], self.pixel_layer[i], self.pixel_pitch, heights[i], precision=self.precision, target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list

        self.seed = seed
        self.last_position = np.zeros(self.n_phasescreens, dtype=self.dtype)
        self.scale_coeff = 1.0

        if self.seed <= 0:
            raise ValueError('seed must be >0')

        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.compute()

    def compute(self):
        # Phase screens list
        self.phasescreens = []
        self.phasescreens_sizes = []

        self.pixel_phasescreens = int(self.xp.max(self.pixel_layer))
        temp_screens = []

        if len(self.xp.unique(self.L0)) == 1:
            # Number of rectangular phase screens from a single square phasescreen
            n_ps_from_square_ps = self.xp.floor(self.pixel_square_phasescreens / self.pixel_phasescreens)
            # Number of square phasescreens
            n_ps = self.xp.ceil(float(self.n_phasescreens) / n_ps_from_square_ps)

            # Seed vector
            seed = self.xp.arange(self.seed, self.seed + int(n_ps))

            # Square phasescreens
            if hasattr(self.L0, '__len__'):
                L0 = self.L0[0]
            else:
                L0 = self.L0
            L0 = np.array([L0])
            square_phasescreens = phasescreens_manager(L0, self.pixel_square_phasescreens,
                                                        self.pixel_pitch, self.data_dir,
                                                        seed=seed, precision=self.precision,
                                                        verbose=self.verbose, xp=self.xp)

            square_ps_index = -1
            ps_index = 0

            for i in range(self.n_phasescreens):
                # Increase square phase-screen index
                if i % n_ps_from_square_ps == 0:
                    square_ps_index += 1
                    ps_index = 0

                temp_screen = square_phasescreens[square_ps_index][int(self.pixel_phasescreens) * ps_index:
                                                                    int(self.pixel_phasescreens) * (ps_index + 1), :]
                temp_screens.append(temp_screen)
                ps_index += 1

        else:
            seed = self.seed + self.xp.arange(self.n_phasescreens)

            if len(seed) != len(self.L0):
                raise ValueError('Number of elements in seed and L0 must be the same!')

            # Square phasescreens
            square_phasescreens = phasescreens_manager(self.L0, self.pixel_square_phasescreens,
                                                        self.pixel_pitch, self.data_dir,
                                                        seed=seed, precision=self.precision,
                                                        verbose=self.verbose, xp=self.xp)

            for i in range(self.n_phasescreens):
                temp_screen = square_phasescreens[i][ :int(self.pixel_phasescreens), :]
                temp_screens.append(temp_screen)


        # Normalize all phasescreens

        for i, temp_screen in enumerate(temp_screens):

            temp_screen = self.to_xp(temp_screen, dtype=self.dtype)
            temp_screen *= self.xp.sqrt(self.Cn2[i])
            temp_screen -= self.xp.mean(temp_screen)

            # Convert to nm
            temp_screen *= ATMO_WAVELENGTH / (2 * np.pi)

            # Flip x-axis for each odd phase-screen
            if i % 2 != 0:
                temp_screen = self.xp.flip(temp_screen, axis=1)

            self.phasescreens.append(temp_screen)
            self.phasescreens_sizes.append(temp_screen.shape[1])

        self.phasescreens_sizes_array = np.asarray(self.phasescreens_sizes)
    
    def setup(self):
        super().setup()
    
        # check that seeing is a 1-element array
        if len(self.local_inputs['seeing'].value) != 1:
            raise ValueError('Seeing input must be a 1-element array')
        
        # Check that wind speed and direction have the correct length
        if len(self.local_inputs['wind_speed'].value) != self.n_phasescreens:
            raise ValueError('Wind speed input must be a {self.n_phasescreens}-elements array')
        if len(self.local_inputs['wind_direction'].value) != self.n_phasescreens:
            raise ValueError('Wind direction input must be a {self.n_phasescreens}-elements array')
        
    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_time = self.t_to_seconds(self.current_time - self.last_t) + self.extra_delta_time
        seeing = float(cpuArray(self.local_inputs['seeing'].value[0]))
        if seeing > 0:
            r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
            self.scale_coeff = (self.pixel_pitch / r0)**(5./6.)
        else:
            self.scale_coeff = 0.0

    def trigger_code(self):

        # if len(self.phasescreens) != len(wind_speed) or len(self.phasescreens) != len(wind_direction):
        #     raise ValueError('Error: number of elements of wind speed and/or direction does not match the number of phasescreens')
        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)
        # Compute the delta position in pixels
        delta_position =  wind_speed * self.delta_time / self.pixel_pitch  # [pixel]
        new_position = self.last_position + delta_position
        # Get quotient and remainder
        wdf, wdi = np.modf(wind_direction/90.0)
        wdf_full = wdf * 90

        if self.cycle_screens:
            new_position = np.where(new_position + self.pixel_layer >= self.phasescreens_sizes_array,  0, new_position)
        new_position_quo = np.floor(new_position).astype(np.int64)
        new_position_rem = (new_position - new_position_quo).astype(self.dtype)
        
        for ii, p in enumerate(self.phasescreens):
            pos = int(new_position_quo[ii])
            ipli = int(self.pixel_layer[ii])
            ipli_p = int(pos + self.pixel_layer[ii])
            layer_phase = (1.0 - new_position_rem[ii]) * p[0: ipli, pos: ipli_p] + new_position_rem[ii] * p[0: ipli, pos+1: ipli_p+1]
            layer_phase = self.xp.rot90(layer_phase, wdi[ii])
            if not wdf_full[ii] == 0:
                layer_phase = self.ndimage_rotate(layer_phase, wdf_full[ii], reshape=False, order=1)
            self.layer_list[ii].phaseInNm[:] = layer_phase * self.scale_coeff
            self.layer_list[ii].generation_time = self.current_time

        # print(f'Phasescreen_shift: {new_position=}') # Verbose?
        # Update position output
        self.last_position = new_position
        self.last_t = self.current_time



