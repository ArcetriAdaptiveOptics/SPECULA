from specula import cpuArray, ASEC2RAD, np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.finite_phase_screen import FinitePhaseScreen
from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen


# Phasescreens are always defined at 500 nm
ATMO_WAVELENGTH = 500.0


class AtmoEvolution(BaseProcessingObj):
    """
    Atmospheric turbulence evolution processing object.
    Generates and evolves atmospheric phase screens based on input parameters.
    Supports both finite (pre-generated and cycled) and infinite (autoregressive) phase screens.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 L0: list,
                 heights: list,
                 Cn2: list,
                 data_dir: str = "",
                 fov: float=0.0,
                 pixel_phasescreens: int=8192,
                 seed: int=1,
                 extra_delta_time: float=0,
                 verbose: bool=False,
                 fov_in_m: float=None,
                 pupil_position: list=None,
                 infinite_ps: bool=False,
                 stencil_size_factor: int=1,
                 target_device_idx: int=None,
                 precision: int=None):
        """
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
            Size of the square phase screens in pixels (used for finite screens). Default is 8192.
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
        infinite_ps : bool, optional
            If True, uses the Infinite Phase Screen model. Default is False.
        stencil_size_factor : int, optional
            Multiplier for the stencil size used in the infinite phase screen model. Default is 1.
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if seed <= 0:
            raise ValueError('seed must be >0')

        self.simul_params = simul_params
        self.infinite_ps = infinite_ps
        self.stencil_size_factor = stencil_size_factor

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.zenithAngleInDeg = self.simul_params.zenithAngleInDeg

        self.n_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_phasescreens, dtype=self.dtype)
        self.last_effective_position = cpuArray(np.zeros(self.n_phasescreens, dtype=self.dtype))
        self.last_t = 0
        self.cycle_screens = True
        self.delta_time = None

        if not hasattr(extra_delta_time,"__len__"):
            self.extra_delta_time = cpuArray(self.n_phasescreens*[extra_delta_time])
        else:
            self.extra_delta_time = cpuArray(extra_delta_time)

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)

        if pupil_position is None:
            pupil_position = [0, 0]

        if self.zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(self.zenithAngleInDeg), dtype=self.dtype)
            print(f'AtmoEvolution: zenith angle is defined as: {self.zenithAngleInDeg} deg')
            print(f'AtmoEvolution: airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        heights = np.array(heights, dtype=self.dtype)
        self.pupil_distances = heights * self.airmass

        fov_rad = fov * ASEC2RAD
        self.pixel_layer = np.ceil(
            (self.pixel_pupil \
                + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) \
                / self.pixel_pitch \
                + abs(self.pupil_distances) / self.pixel_pitch * fov_rad) / 2.0
        ) * 2.0

        if fov_in_m is not None:
            self.pixel_layer = np.full_like(
                heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2
            )

        self.L0 = L0
        if np.isscalar(self.L0):
            self.L0 = [self.L0] * self.n_phasescreens
        elif len(self.L0) != self.n_phasescreens:
            raise ValueError(f"L0 must have the same length as heights ({self.n_phasescreens}),"
                             f" got {len(self.L0)}")

        self.Cn2 = np.array(Cn2, dtype=self.dtype)

        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

        self.pixel_pupil = self.pixel_pupil
        self.data_dir = data_dir

        self.pixel_square_phasescreens = pixel_phasescreens

        # Only check this limit for finite screens
        if not self.infinite_ps and self.pixel_square_phasescreens < max(self.pixel_layer):
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')

        self.verbose = verbose

        # Initialize layer list with correct heights
        self.layer_list = []
        for i in range(self.n_phasescreens):
            layer = Layer(self.pixel_layer[i],
                          self.pixel_layer[i],
                          self.pixel_pitch, heights[i],
                          precision=self.precision,
                          target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list

        self.phasescreens = []
        self.phasescreens_sizes = []
        self.pixel_phasescreens = None
        self.phasescreens_sizes_array = None

        # This array unifies the per-layer math in _update_layer_list
        self.layer_scale = self.xp.zeros(self.n_phasescreens, dtype=self.dtype)

        self.seed = seed
        self.scale_coeff = 1.0


    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.compute()

    def compute(self):
        self.phasescreens = []
        self.phasescreens_sizes = []

        if self.infinite_ps:
            self._compute_infinite()
        else:
            self._compute_finite()

    def _compute_infinite(self):
        print('Creating infinite phase screens..')

        # For infinite screens, Cn2 and wavelength conversion happen at runtime
        self.layer_scale = self.xp.asarray(np.sqrt(self.Cn2), dtype=self.dtype)

        seeds = self.seed + self.xp.arange(self.n_phasescreens)
        # Compute reference r0 at 500 nm for the given seeing, which will be used
        # in the computation of the infinite screens
        seeing = 1.0
        self.ref_r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
        self.ref_r0 *= (ATMO_WAVELENGTH / 500.0 )**(6./5.)

        for i in range(self.n_phasescreens):
            if self.verbose:
                print(f'Creating {i}-th infinite phase screen')
                print(f'    r0: {self.ref_r0}, L0: {self.L0[i]}, size: {self.pixel_layer[i]}')

            temp_infinite_screen = InfinitePhaseScreen(mx_size=self.pixel_layer[i],
                                                       pixel_scale=self.pixel_pitch,
                                                       r0=self.ref_r0,
                                                       L0=self.L0[i],
                                                       random_seed=int(seeds[i]),
                                                       stencil_size_factor=self.stencil_size_factor,
                                                       target_device_idx=self.target_device_idx,
                                                       precision=self.precision)
            self.phasescreens.append(temp_infinite_screen)
            self.phasescreens_sizes.append(self.pixel_layer[i])

        self.phasescreens_sizes_array = np.asarray(self.phasescreens_sizes)

    def _compute_finite(self):
        self.pixel_phasescreens = int(self.xp.max(self.pixel_layer))
        temp_screens = []

        # The finite screens bake in the Cn2 and wavelength conversion here
        self.layer_scale = self.xp.ones(self.n_phasescreens, dtype=self.dtype)

        if len(self.xp.unique(self.L0)) == 1:
            n_ps_from_square_ps = self.xp.floor(
                self.pixel_square_phasescreens / self.pixel_phasescreens)
            n_ps = self.xp.ceil(float(self.n_phasescreens) / n_ps_from_square_ps)
            seed = self.xp.arange(self.seed, self.seed + int(n_ps))

            L0_val = self.L0[0] if hasattr(self.L0, '__len__') else self.L0
            L0_arr = np.array([L0_val])

            square_phasescreens = phasescreens_manager(L0_arr, self.pixel_square_phasescreens,
                                                        self.pixel_pitch, self.data_dir,
                                                        seed=seed, precision=self.precision,
                                                        verbose=self.verbose, xp=self.xp)

            square_ps_index = -1
            ps_index = 0

            for i in range(self.n_phasescreens):
                if i % n_ps_from_square_ps == 0:
                    square_ps_index += 1
                    ps_index = 0

                temp_screen = square_phasescreens[square_ps_index][
                    int(self.pixel_phasescreens) * ps_index:
                    int(self.pixel_phasescreens) * (ps_index + 1), :
                ]
                temp_screens.append(temp_screen)
                ps_index += 1
        else:
            seed = self.seed + self.xp.arange(self.n_phasescreens)
            square_phasescreens = phasescreens_manager(self.L0,
                                                       self.pixel_square_phasescreens,
                                                       self.pixel_pitch,
                                                       self.data_dir,
                                                       seed=seed,
                                                       precision=self.precision,
                                                       verbose=self.verbose,
                                                       xp=self.xp)

            for i in range(self.n_phasescreens):
                temp_screen = square_phasescreens[i][ :int(self.pixel_phasescreens), :]
                temp_screens.append(temp_screen)


        # Normalize and instantiate FinitePhaseScreen objects
        for i, temp_screen in enumerate(temp_screens):
            temp_screen = self.to_xp(temp_screen, dtype=self.dtype)
            temp_screen *= self.xp.sqrt(self.Cn2[i])
            temp_screen -= self.xp.mean(temp_screen)
            temp_screen *= ATMO_WAVELENGTH / (2 * np.pi)

            # Flip x-axis for each odd phase-screen
            if i % 2 != 0:
                temp_screen = self.xp.flip(temp_screen, axis=1)

            finite_screen = FinitePhaseScreen(
                full_screen=temp_screen,
                target_device_idx=self.target_device_idx,
                precision=self.precision
            )

            self.phasescreens.append(finite_screen)
            self.phasescreens_sizes.append(finite_screen.width)

        self.phasescreens_sizes_array = np.asarray(self.phasescreens_sizes)

    def setup(self):
        super().setup()

        if len(self.local_inputs['seeing'].value) != 1:
            raise ValueError('Seeing input must be a 1-element array')

        if len(self.local_inputs['wind_speed'].value) != self.n_phasescreens:
            raise ValueError(f'Wind speed input must be a {self.n_phasescreens}-elements array')
        if len(self.local_inputs['wind_direction'].value) != self.n_phasescreens:
            raise ValueError(f'Wind direction input must be a {self.n_phasescreens}-elements array')


    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_time = cpuArray(
            self.n_phasescreens*[self.t_to_seconds(self.current_time - self.last_t)]
        )
        seeing = float(cpuArray(self.local_inputs['seeing'].value[0]))

        if seeing > 0:
            if self.infinite_ps:
                r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
                r0 *= (ATMO_WAVELENGTH / 500)**(6./5.)
                scale_r0 = (self.ref_r0 / r0)**(5./6.)

                scale_wvl = ATMO_WAVELENGTH / (2 * np.pi)
                self.scale_coeff = scale_r0 * scale_wvl
            else:
                # Finite screens scale by pixel pitch vs r0
                r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
                self.scale_coeff = (self.pixel_pitch / r0)**(5./6.)
        else:
            self.scale_coeff = 0.0

    def trigger_code(self):
        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        delta_position = wind_speed * self.delta_time / self.pixel_pitch  # [pixel]

        new_position, effective_position = self._update_layer_list(
            wind_speed=wind_speed,
            delta_position=delta_position,
            extra_delta_time=self.extra_delta_time,
            last_position=self.last_position,
            layer_list=self.layer_list,
            wind_direction=wind_direction
        )

        self.last_position[:] = new_position
        self.last_effective_position[:] = effective_position
        self.last_t = self.current_time


    def _update_layer_list(self, wind_speed, delta_position, extra_delta_time,
                          last_position, layer_list, wind_direction):
        """Update a layer list with given extra_delta_time."""

        extra_offset = wind_speed * extra_delta_time / self.pixel_pitch
        new_position = last_position + delta_position
        effective_position = new_position + extra_offset

        effective_position_quo = np.floor(effective_position).astype(np.int64)
        effective_position_rem = (effective_position - effective_position_quo).astype(self.dtype)

        # Update each layer using the perfectly unified interface
        for ii, p in enumerate(self.phasescreens):
            pos = int(effective_position_quo[ii])
            out_size = int(self.pixel_layer[ii])
            angle = float(wind_direction[ii])

            phase_0 = p.extract_phase(shift_step=pos, angle_deg=angle, output_size=out_size)
            phase_1 = p.extract_phase(shift_step=pos + 1, angle_deg=angle, output_size=out_size)

            layer_phase = (1.0 - effective_position_rem[ii]) * phase_0 \
                        + effective_position_rem[ii] * phase_1

            # Unified scaling: layer_scale holds 1.0 for finite, or Cn2*wvl scaling for infinite
            layer_list[ii].phaseInNm[:] = layer_phase * self.scale_coeff * self.layer_scale[ii]
            layer_list[ii].generation_time = self.current_time

        last_position[:] = new_position

        return new_position, effective_position
