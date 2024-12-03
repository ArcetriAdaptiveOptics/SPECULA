import numpy as np
from specula import show_in_profiler

from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.base_list import BaseList
from specula.data_objects.layer import Layer
from specula.lib.cv_coord import cv_coord
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.connections import InputValue
from specula import cpuArray

class AtmoEvolution(BaseProcessingObj):
    def __init__(self, L0, pixel_pitch, heights, Cn2, pixel_pupil, data_dir, source_dict, wavelengthInNm: float=500.0,
                 zenithAngleInDeg=None, mcao_fov=None, pixel_phasescreens=None, seed: int=1, target_device_idx=None, precision=None,
                 verbose=None, user_defined_phasescreen: str='', force_mcao_fov=False, make_cycle=None,
                 fov_in_m=None, pupil_position=None):


        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self.n_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_phasescreens)
        self.last_t = 0
        self.extra_delta_time = 0
        self.cycle_screens = True

        self.delta_time = 1
        self.seeing = 1
        self.wind_speed = 1
        self.wind_direction = 1
        self.airmass = 1
        self.wavelengthInNm = wavelengthInNm
        self.pixel_pitch = pixel_pitch         
        
        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)

        if pupil_position is None:
            pupil_position = [0, 0]
        
        if zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg), dtype=self.dtype)
            print(f'Atmo_Evolution: zenith angle is defined as: {zenithAngleInDeg} deg')
            print(f'Atmo_Evolution: airmass is: {self.airmass}')
        else:
            self.airmass = np.array(1.0, dtype=self.dtype)
        heights = np.array(heights, dtype=self.dtype) * self.airmass

        # Conversion coefficient from arcseconds to radians
        sec2rad = 4.848e-6
        
        if force_mcao_fov:
            print(f'\nATTENTION: MCAO FoV is forced to diameter={mcao_fov} arcsec\n')
            alpha_fov = mcao_fov / 2.0
        else:
            alpha_fov = 0.0
            for source in source_dict.values():
                alpha_fov = max(alpha_fov, *abs(cv_coord(from_polar=[source.phi, source.r_arcsec],
                                                       to_rect=True, degrees=False, xp=np)))
            if mcao_fov is not None:
                alpha_fov = max(alpha_fov, mcao_fov / 2.0)
        
        # Max star angle from arcseconds to radians
        rad_alpha_fov = alpha_fov * sec2rad

        # Compute layers dimension in pixels
        self.pixel_layer = np.ceil((pixel_pupil + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) / pixel_pitch + 
                               2.0 * abs(heights) / pixel_pitch * rad_alpha_fov) / 2.0) * 2.0
        if fov_in_m is not None:
            self.pixel_layer = np.full_like(heights, int(fov_in_m / pixel_pitch / 2.0) * 2)
        
        self.L0 = L0
        self.heights = heights
        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.pixel_pupil = pixel_pupil
        self.data_dir = data_dir
        self.make_cycle = make_cycle
        self.seeing = None
        self.wind_speed = None
        self.wind_direction = None

        if pixel_phasescreens is None:
            self.pixel_square_phasescreens = 8192
        else:
            self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < max(self.pixel_layer):
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')
        
        self.verbose = verbose if verbose is not None else False

        # Use a specific user-defined phase screen if provided
        if user_defined_phasescreen is not None:
            self.user_defined_phasescreen = user_defined_phasescreen
        
        # Initialize layer list with correct heights
        self.layer_list = BaseList(target_device_idx=self.target_device_idx)
        for i in range(self.n_phasescreens):
            layer = Layer(self.pixel_layer[i], self.pixel_layer[i], pixel_pitch, heights[i], precision=self.precision, target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list
        
        if seed is not None:
            self.seed = seed
        self.last_position = np.zeros(self.n_phasescreens, dtype=self.dtype)

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

        if self.user_defined_phasescreen:
            temp_screen = fits.getdata(self.user_defined_phasescreen)

            if len(self.Cn2) > 1:
                raise ValueError('The user-defined phasescreen works only if the total phasescreens are 1.')

            if temp_screen.shape[0] < temp_screen.shape[1]:
                temp_screen = temp_screen.T

            temp_screen -= self.xp.mean(temp_screen)
            # Convert to nm
            temp_screen *= self.wavelengthInNm / (2 * self.xp.pi)
            
            self.phasescreens.append(temp_screen)
            self.phasescreens_sizes.append(temp_screen.shape[1])

        else:
            self.pixel_phasescreens = self.xp.max(self.pixel_layer)

            if len(self.xp.unique(self.L0)) == 1:
                # Number of rectangular phase screens from a single square phasescreen
                n_ps_from_square_ps = self.xp.floor(self.pixel_square_phasescreens / self.pixel_phasescreens)
                # Number of square phasescreens
                n_ps = self.xp.ceil(float(self.n_phasescreens) / n_ps_from_square_ps)

                # Seed vector
                seed = self.xp.arange(self.seed, self.seed + int(n_ps))

                # Square phasescreens
                if self.make_cycle:
                    pixel_square_phasescreens = self.pixel_square_phasescreens - self.pixel_pupil
                    ps_cycle = get_layers(1, pixel_square_phasescreens, pixel_square_phasescreens * self.pixel_pitch,
                                          500e-9, 1, L0=self.L0[0], par=par, START=start, SEED=seed, DIR=self.data_dir,
                                          FILE=filename, no_sha=True, verbose=self.verbose)
                    ps_cycle = self.xp.vstack([ps_cycle, ps_cycle[:, :self.pixel_pupil]])
                    ps_cycle = self.xp.hstack([ps_cycle, ps_cycle[:self.pixel_pupil, :]])

                    square_phasescreens = [ps_cycle * 4 * self.xp.pi]  # 4 * π is added to get the correct amplitude
                else:
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

                    temp_screen = self.xp.array(square_phasescreens[square_ps_index][int(self.pixel_phasescreens) * ps_index:
                                                                       int(self.pixel_phasescreens) * (ps_index + 1), :], dtype=self.dtype)
                    # print('self.Cn2[i]', self.Cn2[i], type(self.Cn2[i]), type(self.Cn2))  # Verbose?
                    # print('temp_screen', temp_screen, type(temp_screen))  # Verbose?

                    temp_screen *= self.xp.sqrt(self.Cn2[i])
                    temp_screen -= self.xp.mean(temp_screen)
                    # Convert to nm
                    temp_screen *= self.wavelengthInNm / (2 * np.pi)

                    temp_screen = self.xp.array(temp_screen, dtype=self.dtype)

                    # Flip x-axis for each odd phase-screen
                    if i % 2 != 0:
                        temp_screen = self.xp.flip(temp_screen, axis=1)

                    ps_index += 1

                    self.phasescreens.append(temp_screen)
                    self.phasescreens_sizes.append(temp_screen.shape[1])


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
                    temp_screen = square_phasescreens[i][:, :self.pixel_phasescreens]
                    temp_screen *= self.xp.sqrt(self.Cn2[i])
                    temp_screen -= self.xp.mean(temp_screen)
                    # Convert to nm
                    temp_screen *= self.wavelengthInNm / (2 * self.xp.pi)
                    self.phasescreens.append(temp_screen)
                    self.phasescreens_sizes.append(temp_screen.shape[1])

        self.phasescreens_sizes_array = np.asarray(self.phasescreens_sizes)
    
#        for p in self.phasescreens:
        self.phasescreens_array = self.xp.asarray(self.phasescreens)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_time = self.t_to_seconds(self.current_time - self.last_t) + self.extra_delta_time        
    
    @show_in_profiler('atmo_evolution.trigger_code')
    def trigger_code(self):
        # if len(self.phasescreens) != len(wind_speed) or len(self.phasescreens) != len(wind_direction):
        #     raise ValueError('Error: number of elements of wind speed and/or direction does not match the number of phasescreens')
        seeing = cpuArray(self.local_inputs['seeing'].value)
        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)
        r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.) # if seeing > 0 else 0.0
        r0wavelength = r0 * (self.wavelengthInNm / 500.0)**(6./5.)
        scale_coeff = (self.pixel_pitch / r0wavelength)**(5./6.) # if seeing > 0 else 0.0
        # Compute the delta position in pixels
        delta_position =  wind_speed * self.delta_time / self.pixel_pitch  # [pixel]
        new_position = self.last_position + delta_position
        # Get quotient and remainder
        new_position_quo = np.floor(new_position).astype(np.int64)
        new_position_rem = (new_position - new_position_quo).astype(self.dtype)
        wdf, wdi = np.modf(wind_direction/90.0)
        wdf_full, wdi_full = np.modf(wind_direction)
        # Check if we need to cycle the screens
        # print(ii, new_position[ii], self.pixel_layer[ii], p.shape[1]) # Verbose?
        if self.cycle_screens:
            new_position = np.where(new_position + self.pixel_layer > self.phasescreens_sizes_array,  0, new_position)
#        for ii, p in enumerate(self.phasescreens):
        #    print(f'phasescreens size: {np.around(p.shape[0], 2)}')
        #    print(f'requested position: {np.around(new_position[ii], 2)}')
        #    raise ValueError(f'phasescreens_shift cannot go out of the {ii}-th phasescreen!')            
        # print(pos, self.pixel_layer) # Verbose?

        for ii, p in enumerate(self.phasescreens):
            pos = int(new_position_quo[ii])
            ipli = int(self.pixel_layer[ii])
            ipli_p = int(pos + self.pixel_layer[ii])
            layer_phase = (1.0 - new_position_rem[ii]) * p[0: ipli, pos: ipli_p] + new_position_rem[ii] * p[0: ipli, pos + 1: ipli_p + 1]
            layer_phase = self.xp.rot90(layer_phase, wdi[ii])
            if not wdf_full[ii]==0:
                layer_phase = self.rotate(layer_phase, wdf_full[ii], reshape=False, order=1)
            self.layer_list[ii].phaseInNm = layer_phase * scale_coeff
            self.layer_list[ii].generation_time = self.current_time

        # print(f'Phasescreen_shift: {new_position=}') # Verbose?
        # Update position output
        self.last_position = new_position
        self.layer_list.generation_time = self.current_time
        self.last_t = self.current_time
        
    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['INTRLVD'] = int(self.interleave)
        hdr['PUPD_TAG'] = self.pupdata_tag
        super().save(filename, hdr)

        with fits.open(filename, mode='append') as hdul:
            hdul.append(fits.ImageHDU(data=self.phasescreens))

    def read(self, filename):
        super().read(filename)
        self.phasescreens = fits.getdata(filename, ext=1)

    def set_last_position(self, last_position):
        self.last_position = last_position

    def set_last_t(self, last_t):
        self.last_t = last_t

    def run_check(self, time_step):
        self.prepare_trigger(0)

        errmsg = ''
        if not (self.seed > 0):
            errmsg += ' Seed <= 0.'
        if not isinstance(self.seeing, BaseValue):
            errmsg += ' Missing input seeing.'
        if not isinstance(self.wind_direction, BaseValue):
            errmsg += ' Missing input wind direction.'
        if not isinstance(self.wind_speed, BaseValue):
            errmsg += ' Missing input speed.'
        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            errmsg += f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.'

        seeing = self.inputs['seeing'].get(self.target_device_idx)
        wind_speed = self.inputs['wind_speed'].get(self.target_device_idx)
        wind_direction = self.inputs['wind_direction'].get(self.target_device_idx)
                
        check = self.seed > 0 and isinstance(seeing, BaseValue) and isinstance(wind_direction, BaseValue) and isinstance(wind_speed, BaseValue)
        if not check:
            raise ValueError(errmsg)
          
        # super().build_stream()
        return check

