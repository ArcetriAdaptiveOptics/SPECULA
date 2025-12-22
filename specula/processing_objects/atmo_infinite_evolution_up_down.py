from specula import cpuArray, np
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams


class AtmoInfiniteEvolutionUpDown(AtmoInfiniteEvolution):
    """Atmospheric infinite phase screens evolution with separate layer lists
    for upward and downward propagation.
    
    This class extends AtmoInfiniteEvolution to provide two independent layer
    lists with different extra_delta_time values. Due to the stateful nature
    of InfinitePhaseScreen.add_line(), temporary copies are made during trigger.
    
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
    extra_delta_time_down : float or list, optional
        Extra time offset for downward propagation in seconds. Default is 0.
    extra_delta_time_up : float or list, optional
        Extra time offset for upward propagation in seconds. Default is 0.
    fov : float, optional
        Field of view in arcseconds. Default is 0.0.
    seed : int, optional
        Seed for random number generation. Must be >0. Default is 1.
    verbose : bool, optional
        If True, enables verbose output. Default is False.
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
                 L0: list = [1.0],
                 heights: list = [0.0],
                 Cn2: list = [1.0],
                 extra_delta_time_down: float = 0,
                 extra_delta_time_up: float = 0,
                 fov: float = 0.0,
                 seed: int = 1,
                 verbose: bool = False,
                 fov_in_m: float = None,
                 pupil_position: list = [0, 0],
                 target_device_idx: int = None,
                 precision: int = None):

        # Initialize with down extra_delta_time
        super().__init__(
            simul_params=simul_params,
            L0=L0,
            heights=heights,
            Cn2=Cn2,
            fov=fov,
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
            self.extra_delta_time_down = cpuArray(
                self.n_infinite_phasescreens * [extra_delta_time_down]
            )
        else:
            self.extra_delta_time_down = cpuArray(extra_delta_time_down)

        if not hasattr(extra_delta_time_up, "__len__"):
            self.extra_delta_time_up = cpuArray(
                self.n_infinite_phasescreens * [extra_delta_time_up]
            )
        else:
            self.extra_delta_time_up = cpuArray(extra_delta_time_up)

        # Set parent's extra_delta_time to down
        self.extra_delta_time = self.extra_delta_time_down

        # Create separate layer lists
        self.layer_list_down = self.layer_list
        self.layer_list_up = []

        for i in range(self.n_infinite_phasescreens):
            layer = Layer(
                self.pixel_layer_size[i],
                self.pixel_layer_size[i],
                self.pixel_pitch,
                heights[i],
                precision=self.precision,
                target_device_idx=self.target_device_idx
            )
            self.layer_list_up.append(layer)

        # Update outputs
        self.outputs['layer_list_down'] = self.layer_list_down
        self.outputs['layer_list_up'] = self.layer_list_up

        # Separate tracking for up
        self.last_position_up = np.zeros(self.n_infinite_phasescreens, dtype=self.dtype)
        self.last_effective_position_up = np.zeros(
            self.n_infinite_phasescreens, dtype=self.dtype
        )
        self.acc_rows_up = np.zeros(self.n_infinite_phasescreens)
        self.acc_cols_up = np.zeros(self.n_infinite_phasescreens)

    def trigger_code(self):
        """Update both lists by saving/restoring phase screen state."""

        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        delta_position = wind_speed * self.delta_time / self.pixel_pitch

        # Determine which direction to process first based on extra_delta_time
        # Process the one with smaller extra_delta_time first (earlier in time)
        down_first = np.all(self.extra_delta_time_down <= self.extra_delta_time_up)

        if down_first:
            # Process down first, save state, then process up
            self._process_propagation_direction(
                wind_speed, wind_direction, delta_position,
                self.extra_delta_time_down, self.last_position,
                self.last_effective_position, self.acc_rows, self.acc_cols,
                self.layer_list_down
            )

            # Save state
            saved_states = self._save_phase_screen_states()

            # Process up
            self._process_propagation_direction(
                wind_speed, wind_direction, delta_position,
                self.extra_delta_time_up, self.last_position_up,
                self.last_effective_position_up, self.acc_rows_up, self.acc_cols_up,
                self.layer_list_up
            )

            # Restore down state (keep phase screens at down position)
            self._restore_phase_screen_states(saved_states)
        else:
            # Process up first, save state, then process down
            self._process_propagation_direction(
                wind_speed, wind_direction, delta_position,
                self.extra_delta_time_up, self.last_position_up,
                self.last_effective_position_up, self.acc_rows_up, self.acc_cols_up,
                self.layer_list_up
            )

            # Save state
            saved_states = self._save_phase_screen_states()

            # Process down
            self._process_propagation_direction(
                wind_speed, wind_direction, delta_position,
                self.extra_delta_time_down, self.last_position,
                self.last_effective_position, self.acc_rows, self.acc_cols,
                self.layer_list_down
            )

            # Restore up state (keep phase screens at up position)
            # Actually, we want to keep them at down, so restore anyway
            # No, we should keep the "later" state for next frame
            # Let's always keep down state
            if not down_first:
                pass  # Already at down state

        self.last_t = self.current_time

    def _save_phase_screen_states(self):
        """Save current state of all phase screens."""
        saved = []
        for ps in self.infinite_phasescreens:
            saved.append({
                'full_scrn': ps.full_scrn.copy(),
                'random_data_col': ps.random_data_col.copy() \
                    if ps.random_data_col is not None else None,
                'random_data_row': ps.random_data_row.copy() \
                    if ps.random_data_row is not None else None
            })
        return saved

    def _restore_phase_screen_states(self, saved_states):
        """Restore phase screens to saved state."""
        for i, ps in enumerate(self.infinite_phasescreens):
            ps.full_scrn = saved_states[i]['full_scrn']
            ps.random_data_col = saved_states[i]['random_data_col']
            ps.random_data_row = saved_states[i]['random_data_row']

    def _process_propagation_direction(self, wind_speed, wind_direction,
                                       delta_position, extra_delta_time,
                                       last_position, last_effective_position,
                                       acc_rows, acc_cols, layer_list):
        """Process one propagation direction (up or down)."""

        extra_offset = wind_speed * extra_delta_time / self.pixel_pitch
        effective_position = last_position + delta_position + extra_offset
        effective_delta_position = effective_position - last_effective_position

        eps = 1e-4

        for ii, phase_screen in enumerate(self.infinite_phasescreens):
            w_y_comp = np.cos(2 * np.pi * wind_direction[ii] / 360.0)
            w_x_comp = np.sin(2 * np.pi * wind_direction[ii] / 360.0)

            frac_rows, rows_to_add = np.modf(
                effective_delta_position[ii] * w_y_comp + acc_rows[ii]
            )
            sr = int(np.sign(rows_to_add))

            frac_cols, cols_to_add = np.modf(
                effective_delta_position[ii] * w_x_comp + acc_cols[ii]
            )
            sc = int(np.sign(cols_to_add))

            # Add integer lines
            if np.abs(w_y_comp) > eps:
                for r in range(int(np.abs(rows_to_add))):
                    phase_screen.add_line(1, sr)
            if np.abs(w_x_comp) > eps:
                for r in range(int(np.abs(cols_to_add))):
                    phase_screen.add_line(0, sc)

            phase_screen0_all = phase_screen.scrnRawAll.copy()
            phase_screen0 = phase_screen.scrnRaw.copy()

            # Fractional interpolation
            srf = int(np.sign(frac_rows))
            scf = int(np.sign(frac_cols))

            if np.abs(frac_rows) > eps:
                phase_screen.add_line(1, srf, False)
            if np.abs(frac_cols) > eps:
                phase_screen.add_line(0, scf, False)

            phase_screen1 = phase_screen.scrnRaw
            interpfactor = np.sqrt(frac_rows**2 + frac_cols**2)
            layer_phase = interpfactor * phase_screen1 + (1.0 - interpfactor) * phase_screen0

            phase_screen.full_scrn = phase_screen0_all
            acc_rows[ii] = frac_rows
            acc_cols[ii] = frac_cols

            layer_list[ii].field[:] = self.xp.stack((layer_phase, layer_phase))
            layer_list[ii].phaseInNm *= self.scale_coeff * self.xp.sqrt(self.Cn2[ii])
            layer_list[ii].A = 1
            layer_list[ii].generation_time = self.current_time

        # Update positions
        last_position[:] = last_position + delta_position
        last_effective_position[:] = effective_position
