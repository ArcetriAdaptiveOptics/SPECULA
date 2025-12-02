import numpy as np
from specula import show_in_profiler

from specula.base_processing_obj import BaseProcessingObj
from specula.base_data_obj import BaseDataObj
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.connections import InputValue
from specula import cpuArray, ASEC2RAD, RAD2ASEC
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen


class AtmoInfiniteEvolution(BaseProcessingObj):
    """Atmospheric infinite phase screens evolution..
    Generates and evolves atmospheric phase screens based on input parameters such as
    seeing, wind speed, and wind direction.

    Note
    ----
    Phase screens are always generated at a reference wavelength of 500 nm.
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object containing global simulation settings.
    L0 : list
        Outer scale(s) of turbulence for each layer in meters.
    heights : list
        Heights of the atmospheric layers in meters (at zenith).
    Cn2 : list
        Fractional Cn2 values for each layer (must sum to 1.0).
    data_dir : str
        Directory path for storing/loading phase screen data (automatically set by simul.py).
    fov : float, optional
        Field of view in arcseconds. Default is 0.0.
    pixel_phasescreens : int, optional
        Size of the square phase screens in pixels. Default is 8192.
    seed : int, optional
        Seed for random number generation. Must be >0. Default is 1.
    extra_delta_time : float or list, optional
        Extra time offset for phase screen evolution in seconds. Default is 0.
    verbose : bool, optional
        If True, enables verbose output during phase screen generation. Default is False.
    fov_in_m : float, optional
        Field of view in meters. If provided, overrides fov parameter. Default is None.
    pupil_position : list, optional
        [x, y] position of the pupil in meters. Default is [0, 0].
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None (uses global setting).
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None (uses global setting).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 L0: list=[1.0],
                 heights: list=[0.0],
                 Cn2: list=[1.0],
                 fov: float=0.0,
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

        self.n_infinite_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_infinite_phasescreens, dtype=self.dtype)
        self.last_effective_position = np.zeros(self.n_infinite_phasescreens, dtype=self.dtype)
        self.last_t = 0
        self.delta_time = None
        # fixed at generation time, then is a input -> rescales the screen?
        self.seeing = 1.0
        self.airmass = 1
        self.ref_wavelengthInNm = 500

        if not hasattr(extra_delta_time,"__len__"):
            self.extra_delta_time = cpuArray(self.n_infinite_phasescreens*[extra_delta_time])
        else:
            self.extra_delta_time = cpuArray(extra_delta_time)

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)

        if pupil_position is None:
            pupil_position = [0, 0]

        if self.zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(self.zenithAngleInDeg), dtype=self.dtype)
            print(f'AtmoInfiniteEvolution: zenith angle is defined as:'
                  f' {self.zenithAngleInDeg} deg')
            print(f'AtmoInfiniteEvolution: airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        heights = np.array(heights, dtype=self.dtype)

        # distances from the pupil accounting for zenith angle
        self.pupil_distances = heights * self.airmass

        alpha_fov = fov / 2.0

        # Max star angle from arcseconds to radians
        rad_alpha_fov = alpha_fov * ASEC2RAD

        # Compute layers dimension in pixels
        self.pixel_layer_size = np.ceil(
            (self.pixel_pupil \
                + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) \
                / self.pixel_pitch \
                + 2.0 * abs(self.pupil_distances) / self.pixel_pitch * rad_alpha_fov) / 2.0
        ) * 2.0
        if fov_in_m is not None:
            self.pixel_layer_size = np.full_like(
                heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2
            )

        self.L0 = L0

        if np.isscalar(self.L0):
            self.L0 = [self.L0] * len(heights)
        elif len(self.L0) != len(heights):
            raise ValueError(f"L0 must have the same length as heights"
                             f" ({len(heights)}), got {len(self.L0)}")

        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.verbose = verbose if verbose is not None else False

        # Initialize layer list with correct heights
        self.layer_list = []
        for i in range(self.n_infinite_phasescreens):
            layer = Layer(self.pixel_layer_size[i],
                          self.pixel_layer_size[i],
                          self.pixel_pitch, heights[i],
                          precision=self.precision,
                          target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list

        self.initScreens(seed)

        self.scale_coeff = 1.0

        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

    def initScreens(self, seed):
        self.seed = seed
        if self.seed <= 0:
            raise ValueError('seed must be >0')
        # Phase screens list
        self.infinite_phasescreens = []
        seed = self.seed + self.xp.arange(self.n_infinite_phasescreens)
        if len(seed) != len(self.L0):
            raise ValueError('Number of elements in seed and L0 must be the same!')

        self.acc_rows = np.zeros((self.n_infinite_phasescreens))
        self.acc_cols = np.zeros((self.n_infinite_phasescreens))

        # Square infinite_phasescreens
        print('Creating phase screens..')
        for i in range(self.n_infinite_phasescreens):
            self.ref_r0 = 0.9759 * 0.5 / (self.seeing * 4.848) \
                * self.airmass**(-3./5.) # if seeing > 0 else 0.0
            self.ref_r0 *= (self.ref_wavelengthInNm / 500.0 )**(6./5.)
            if self.verbose: # pragma: no cover
                print(f'Creating {i}-th phase screen')
                print(f'    r0: {self.ref_r0}, L0: {self.L0[i]},'
                      f' size: {self.pixel_layer_size[i]}')
            temp_infinite_screen = InfinitePhaseScreen(self.pixel_layer_size[i],
                                                       self.pixel_pitch,
                                                       self.ref_r0,
                                                       self.L0[i],
                                                       random_seed=int(seed[i]),
                                                       xp=self.xp,
                                                       target_device_idx=self.target_device_idx,
                                                       precision=self.precision )
            self.infinite_phasescreens.append(temp_infinite_screen)

    def setup(self):
        super().setup()
        # check that seeing is a 1-element array
        if len(self.local_inputs['seeing'].value) != 1:
            raise ValueError('Seeing input must be a 1-element array')

        # Check that wind speed and direction have the correct length
        if len(self.local_inputs['wind_speed'].value) != self.n_infinite_phasescreens:
            raise ValueError(f'Wind speed input must be a'
                             f' {self.n_infinite_phasescreens}-elements array')
        if len(self.local_inputs['wind_direction'].value) != self.n_infinite_phasescreens:
            raise ValueError(f'Wind direction input must be a'
                             f' {self.n_infinite_phasescreens}-elements array')

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_time = cpuArray(
            self.n_infinite_phasescreens*[self.t_to_seconds(self.current_time - self.last_t)]
        )
        seeing = float(cpuArray(self.local_inputs['seeing'].value[0]))

        if seeing > 0:
            r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
            r0 *= (self.ref_wavelengthInNm / 500)**(6./5.)
            scale_r0 = (self.ref_r0 / r0)**(5./6.)
        else:
            scale_r0 = 0.0

        scale_wvl = self.ref_wavelengthInNm / (2 * np.pi)
        self.scale_coeff = scale_r0 * scale_wvl

    @show_in_profiler('atmo_evolution.trigger_code')
    def trigger_code(self):
        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        # Compute the delta position in pixels
        delta_position =  wind_speed * self.delta_time / self.pixel_pitch  # [pixel]

        # Compute extra offset that doesn't get accumulated
        extra_offset = wind_speed * self.extra_delta_time / self.pixel_pitch  # [pixel]

        # Effective position = accumulated position + constant offset
        # Note: extra_offset is added at each frame because it is a function of wind speed
        effective_position = self.last_position + delta_position + extra_offset  # [pixel]

        # Change in effective position since last frame
        effective_delta_position = effective_position - self.last_effective_position  # [pixel]

        eps = 1e-4

        for ii, phase_screen in enumerate(self.infinite_phasescreens):
            w_y_comp = np.cos(2*np.pi*(wind_direction[ii])/360.0)
            w_x_comp = np.sin(2*np.pi*(wind_direction[ii])/360.0)
            frac_rows, rows_to_add = np.modf(
                effective_delta_position[ii] * w_y_comp + self.acc_rows[ii]
            )
            #sr = int( (np.sign(rows_to_add) + 1) / 2 )
            sr = int(np.sign(rows_to_add) )
            frac_cols, cols_to_add = np.modf(
                effective_delta_position[ii] * w_x_comp + self.acc_cols[ii]
            )
            #sc = int( (-np.sign(cols_to_add) + 1) / 2 )
            sc = int(np.sign(cols_to_add) )
            # print('rows_to_add, cols_to_add', rows_to_add, cols_to_add)
            if np.abs(w_y_comp)>eps:
                for r in range(int(np.abs(rows_to_add))):
                    phase_screen.add_line(1, sr)
            if np.abs(w_x_comp)>eps:
                for r in range(int(np.abs(cols_to_add))):
                    phase_screen.add_line(0, sc)
            phase_screen0_all = phase_screen.scrnRawAll.copy()
            phase_screen0 = phase_screen.scrnRaw.copy()
            # print('w_y_comp, w_x_comp', w_y_comp, w_x_comp)
            # print('frac_rows, frac_cols', frac_rows, frac_cols)
            srf = int(np.sign(frac_rows) )
            scf = int(np.sign(frac_cols) )

            if np.abs(frac_rows)>eps:
                phase_screen.add_line(1, srf, False)
            if np.abs(frac_cols)>eps:
                phase_screen.add_line(0, scf, False)
            phase_screen1 = phase_screen.scrnRaw
            interpfactor = np.sqrt(frac_rows**2 + frac_cols**2 )
            layer_phase = interpfactor * phase_screen1 + (1.0-interpfactor) * phase_screen0
            phase_screen.full_scrn = phase_screen0_all
            self.acc_rows[ii] = frac_rows
            self.acc_cols[ii] = frac_cols
            # print('acc_rows', self.acc_rows)
            # print('acc_cols', self.acc_cols)
            self.layer_list[ii].field[:] = self.xp.stack((layer_phase, layer_phase))
            self.layer_list[ii].phaseInNm *= self.scale_coeff*self.xp.sqrt(self.Cn2[ii])
            self.layer_list[ii].A = 1
            self.layer_list[ii].generation_time = self.current_time

        self.last_position = self.last_position + delta_position
        self.last_effective_position = effective_position.copy()
        self.last_t = self.current_time
