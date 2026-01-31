from specula import cpuArray, np
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams


class AtmoEvolutionUpDown(AtmoEvolution):
    """Atmospheric turbulence evolution with separate layer lists
    for upward and downward propagation.
    
    This class extends AtmoEvolution to provide two independent layer
    lists with different extra_delta_time values, useful for simulating
    satellite laser ranging where uplink and downlink paths experience
    different temporal offsets.
    
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
        Directory path for storing/loading phase screen data.
    extra_delta_time_down : float or list, optional
        Extra time offset for downward propagation in seconds. Default is 0.
    extra_delta_time_up : float or list, optional
        Extra time offset for upward propagation in seconds. Default is 0.
    fov : float, optional
        Field of view in arcseconds. Default is 0.0.
    pixel_phasescreens : int, optional
        Size of the square phase screens in pixels. Default is 8192.
    seed : int, optional
        Seed for random number generation. Must be >0. Default is 1.
    verbose : bool, optional
        If True, enables verbose output during phase screen generation. Default is False.
    fov_in_m : float, optional
        Field of view in meters. If provided, overrides fov parameter. Default is None.
    pupil_position : list, optional
        [x, y] position of the pupil in meters. Default is [0, 0].
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None.
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 L0: list,
                 heights: list,
                 Cn2: list,
                 data_dir: str,
                 extra_delta_time_down: float = 0,
                 extra_delta_time_up: float = 0,
                 fov: float = 0.0,
                 pixel_phasescreens: int = 8192,
                 seed: int = 1,
                 verbose: bool = False,
                 fov_in_m: float = None,
                 pupil_position: list = [0, 0],
                 target_device_idx: int = None,
                 precision: int = None):

        # Initialize parent class with downward extra_delta_time
        super().__init__(
            simul_params=simul_params,
            L0=L0,
            heights=heights,
            Cn2=Cn2,
            data_dir=data_dir,
            fov=fov,
            pixel_phasescreens=pixel_phasescreens,
            seed=seed,
            extra_delta_time=extra_delta_time_down,
            verbose=verbose,
            fov_in_m=fov_in_m,
            pupil_position=pupil_position,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Store both extra_delta_time arrays
        if not hasattr(extra_delta_time_down, "__len__"):
            self.extra_delta_time_down = cpuArray(self.n_phasescreens * [extra_delta_time_down])
        else:
            self.extra_delta_time_down = cpuArray(extra_delta_time_down)

        if not hasattr(extra_delta_time_up, "__len__"):
            self.extra_delta_time_up = cpuArray(self.n_phasescreens * [extra_delta_time_up])
        else:
            self.extra_delta_time_up = cpuArray(extra_delta_time_up)

        # Set the parent's extra_delta_time to down (for compatibility)
        self.extra_delta_time = self.extra_delta_time_down

        # Create separate layer lists for up and down
        self.layer_list_down = self.layer_list
        self.layer_list_up = []

        for i in range(self.n_phasescreens):
            layer = Layer(
                self.pixel_layer[i],
                self.pixel_layer[i],
                self.pixel_pitch,
                heights[i],
                precision=self.precision,
                target_device_idx=self.target_device_idx
            )
            self.layer_list_up.append(layer)

        # Update outputs to include both layer lists
        self.outputs['layer_list_down'] = self.layer_list_down
        self.outputs['layer_list_up'] = self.layer_list_up

        # Track positions for up propagation separately
        self.last_position_up = np.zeros(self.n_phasescreens, dtype=self.dtype)

    def trigger_code(self):
        """Update both downward and upward layer lists with different time offsets."""

        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        # Compute the delta position in pixels (time evolution)
        delta_position = wind_speed * self.delta_time / self.pixel_pitch  # [pixel]

        # Get quotient and remainder for wind direction
        wdf, wdi = np.modf(wind_direction / 90.0)
        wdf_full = wdf * 90

        # Process downward propagation
        new_position_down, effective_position_down = self._update_layer_list(
            wind_speed=wind_speed,
            delta_position=delta_position,
            extra_delta_time=self.extra_delta_time_down,
            last_position=self.last_position,
            layer_list=self.layer_list_down,
            wdi=wdi,
            wdf_full=wdf_full
        )

        # Process upward propagation
        new_position_up, effective_position_up = self._update_layer_list(
            wind_speed=wind_speed,
            delta_position=delta_position,
            extra_delta_time=self.extra_delta_time_up,
            last_position=self.last_position_up,
            layer_list=self.layer_list_up,
            wdi=wdi,
            wdf_full=wdf_full
        )

        # Update tracking
        self.last_position[:] = new_position_down
        self.last_position_up[:] = new_position_up
        self.last_effective_position[:] = effective_position_down
        self.last_t = self.current_time
