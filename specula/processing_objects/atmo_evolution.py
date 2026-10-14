from typing import List
from specula import ASEC2RAD, np, fuse
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.lib.phasescreen_manager import phasescreens_manager
from specula.lib.affine_transform import affine_transform
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams


# Phasescreens are always defined at 500 nm
ATMO_WAVELENGTH = 500.0



@fuse(kernel_name='atmo_seeing_scale')
def _seeing_scale(scale_coef, seeing, seeing_scale_factor, xp):
    """Seeing scale coefficient, zero for seeing <= 0 (no division by seeing)"""
    scale_coef[...] = xp.maximum(seeing, 0) ** (5. / 6.) * seeing_scale_factor


@fuse(kernel_name='atmo_rotation_matrix')
def _rotation_matrix(r00, r01, r02, r10, r11, r12, wind_direction, center, xp):
    """Affine matrix [[a, b, offset0], [-b, a, offset1]] (one element per argument)
    of rot90() by k = trunc(wind_direction / 90) followed by ndimage rotate() by the
    residual angle, both around the layer center: A = R(90 k) @ R(residual).
    cos(90 k) and sin(90 k) are exact, so A is an exact permutation when the
    residual angle is zero."""
    q = wind_direction / 90
    k = xp.trunc(q)
    residual = xp.radians((q - k) * 90)
    m = xp.remainder(k, 4)
    cos_k = (1 - m % 2) * (1 - m)     # 1, 0, -1, 0 for m = 0, 1, 2, 3
    sin_k = (m % 2) * (2 - m)         # 0, 1, 0, -1 for m = 0, 1, 2, 3
    cos_r = xp.cos(residual)
    sin_r = xp.sin(residual)
    a = cos_k * cos_r - sin_k * sin_r
    b = cos_k * sin_r + sin_k * cos_r
    r00[...] = a
    r01[...] = b
    r02[...] = center * (1 - a - b)
    r10[...] = -b
    r11[...] = a
    r12[...] = center * (1 - a + b)


@fuse(kernel_name='atmo_positions')
def _positions(last_position, effective_position, wind_speed, delta_time, extra_delta_time,
               pixel_layer, screen_size, pixel_pitch, cycle_screens, xp):
    """Accumulate the positions, cycling the screens considering the effective position,
    and compute the effective position (accumulated position + constant offset) [pixel]"""
    extra_offset = wind_speed * extra_delta_time / pixel_pitch
    new_position = last_position + wind_speed * delta_time / pixel_pitch
    new_position = xp.where(cycle_screens & (new_position + extra_offset + pixel_layer >= screen_size),
                            0, new_position)
    last_position[...] = new_position
    effective_position[...] = new_position + extra_offset


class AtmoEvolution(BaseProcessingObj):
    """
    Atmospheric turbulence evolution processing object.
    Generates and evolves atmospheric phase screens based on input parameters such as
    seeing, wind speed, and wind direction.

    Each time step is split so that trigger_code() can be captured in a CUDA graph
    (see BaseProcessingObj.build_stream(), called in setup()):

    - prepare_trigger() only updates the device time step, if it has changed.
      trigger_code() reads the seeing, wind speed and wind direction inputs
      directly from local_inputs: their arrays keep the same address at each step
      (the producer's own array on the same device, or a persistent copy
      updated in place when coming from another device).
    - trigger_code() computes on the device the seeing scale coefficient
      (self.scale_coef), the rotation matrices (self.rot_matrix), the layer
      positions (including screen cycling) and window matrices (self.win_matrix),
      using fused kernels. Then each layer is computed with two
      lib.affine_transform.affine_transform() calls (window interpolation, then
      rotation) and a multiplication by the scale coefficient. No host values or
      transfers are
      involved, so the captured graph stays valid when positions, wind or seeing
      change.
    - post_trigger() updates the host-side state: last_t and the generation_time
      of the output layers.

    On CPU, the same code runs with numpy (affine_transform() has fast CPU paths).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 L0: float | List[float],
                 heights: list,
                 Cn2: list,
                 data_dir: str = "",
                 fov: float=0.0,
                 pixel_phasescreens: int=8192,
                 seed: int=1,
                 extra_delta_time: float=0,
                 fov_in_m: float=None,
                 pupil_position:list =[0,0],
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Note
        ----
        Phase screens are always generated at a reference wavelength of 500 nm.

        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing global simulation settings.
        L0 : float or list [m]
            Outer scale(s) of turbulence for each layer in meters.
        heights : list [m]
            Heights of the atmospheric layers in meters (at zenith).
        Cn2 : list [1]
            Fractional Cn2 values for each layer (must sum to 1.0).
        data_dir : str
            Directory path for storing/loading phase screen data (automatically set by simul.py).
        fov : float [arcsec], optional
            Field of view in arcseconds. Default is 0.0.
        pixel_phasescreens : int [1], optional
            Size of the square phase screens in pixels. Default is 8192.
        seed : int [1], optional
            Seed for random number generation. Must be >0. Default is 1.
        extra_delta_time : float or list [s], optional
            Extra time offset for phase screen evolution in seconds. Default is 0.
        fov_in_m : float [m], optional
            Field of view in meters. If provided, overrides fov parameter. Default is None.
        pupil_position : list [m], optional
            [x, y] position of the pupil in meters. Default is [0, 0].
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

        if seed <= 0:
            raise ValueError('seed must be >0')

        self.n_phasescreens = len(heights)
        self.last_position = self.xp.zeros(self.n_phasescreens, dtype=self.dtype)
        self.last_t = 0
        self.cycle_screens = True
        self.delta_time = None

        if not hasattr(extra_delta_time,"__len__"):
            self.extra_delta_time = self.to_xp(self.n_phasescreens*[extra_delta_time],
                                               dtype=self.dtype)
        else:
            self.extra_delta_time = self.to_xp(extra_delta_time, dtype=self.dtype)

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)

        if zenithAngleInDeg is not None:
            airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg), dtype=self.dtype)
            self.logger.info(f'zenith angle is defined as: {zenithAngleInDeg} deg')
            self.logger.info(f'airmass is: {airmass}')
        else:
            airmass = 1.0

        heights = np.array(heights, dtype=self.dtype)
        # distances from the pupil accounting for zenith angle, kept in self for testing purposes
        self.pupil_distances = heights * airmass

        # pixel_pitch / r0 = seeing * const, so that the seeing scale coefficient
        # (pixel_pitch / r0)**(5/6) = seeing**(5/6) * self.seeing_scale_factor
        self.seeing_scale_factor = (self.pixel_pitch * 4.848 / (0.9759 * 0.5)
                                    * float(airmass)**(3./5.))**(5./6.)

        if fov_in_m is not None:
            self.pixel_layer = np.full_like(
                heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2
            )
        else:
            fov_rad = fov * ASEC2RAD
            self.pixel_layer = np.ceil(
                (self.pixel_pupil \
                    + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) \
                    / self.pixel_pitch \
                    + abs(self.pupil_distances) / self.pixel_pitch * fov_rad) / 2.0
            ) * 2.0

        self.L0 = L0
        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.data_dir = data_dir

        self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < max(self.pixel_layer):
            raise ValueError('Error: phase-screens dimension must be'
                             'greater than layer dimension!')

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

        # Layer lists, each with its extra delta time and accumulated position.
        # Derived classes with more layer lists redefine these lists.
        # The arrays used by trigger_code() are allocated in setup().
        self.layer_lists = [self.layer_list]
        self.extra_delta_times = [self.extra_delta_time]
        self.last_positions = [self.last_position]
        # Interpolated (not rotated) layer windows. Each one is used only while computing
        # its layer, so they are contiguous views of a single buffer for the largest layer.
        window_buffer = self.xp.zeros(int(max(self.pixel_layer)) ** 2, dtype=self.dtype)
        self.windows = [window_buffer[:int(n) ** 2].reshape(int(n), int(n)) for n in self.pixel_layer]

        self.seed = seed

        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

        self.compute()

    @classmethod
    def input_names(cls):
        return {'seeing': InputDesc(BaseValue, 'Atmospheric seeing value'),
                'wind_speed': InputDesc(BaseValue, 'Wind speed for each atmospheric layer'),
                'wind_direction': InputDesc(BaseValue, 'Wind direction for each atmospheric layer')}

    @classmethod
    def output_names(cls):
        return {'layer_list': OutputDesc(list, 'List of atmospheric phase screen layers')}

    def compute(self):
        # Phase screens list
        self.phasescreens = []
        phasescreens_sizes = []

        pixel_phasescreens = int(self.xp.max(self.pixel_layer))

        # Each layer is a strip of pixel_phasescreens rows of a square phase screen.
        # With a single L0, several strips are cut from each square phase screen;
        # otherwise each layer uses the first strip of its own square phase screen.
        if len(np.unique(self.L0)) == 1:
            strips_per_square = self.pixel_square_phasescreens // pixel_phasescreens
            square_L0 = np.atleast_1d(self.L0)[:1]
        else:
            if len(self.L0) != self.n_phasescreens:
                raise ValueError('Number of elements in seed and L0 must be the same!')
            strips_per_square = 1
            square_L0 = self.L0
        n_squares = -(-self.n_phasescreens // strips_per_square)

        # Square phasescreens
        square_phasescreens = phasescreens_manager(square_L0, self.pixel_square_phasescreens,
                                                   self.pixel_pitch, self.data_dir,
                                                   seed=self.seed + self.xp.arange(n_squares),
                                                   precision=self.precision, xp=self.xp)
        rows = pixel_phasescreens
        temp_screens = [square_phasescreens[i // strips_per_square][
                            (i % strips_per_square) * rows:(i % strips_per_square + 1) * rows, :]
                        for i in range(self.n_phasescreens)]

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

            # Contiguous, otherwise affine_transform() makes a copy at each call
            self.phasescreens.append(self.xp.ascontiguousarray(temp_screen))
            phasescreens_sizes.append(temp_screen.shape[1])

        self.phasescreens_sizes_array = np.asarray(phasescreens_sizes)

    def setup(self):
        """Allocate the device arrays used by trigger_code(), and capture it.

        - delta_time_xp (scalar): time step [s]
        - scale_coef (1 element): seeing scale coefficient
        - win_matrix (n_layer_lists, n_phasescreens, 2, 3): affine matrix that
          extracts each layer window from its phase screen, with linear interpolation
          at the effective position (the x offset, also self.last_effective_position)
        - rot_matrix (n_phasescreens, 2, 3): affine matrix of the rotation by the
          wind direction around the layer center, equivalent to rot90() followed
          by ndimage rotate()
        """
        super().setup()

        # check that seeing is a 1-element array
        if len(self.local_inputs['seeing'].value) != 1:
            raise ValueError('Seeing input must be a 1-element array')

        # Check that wind speed and direction have the correct length
        if len(self.local_inputs['wind_speed'].value) != self.n_phasescreens:
            raise ValueError('Wind speed input must be a {self.n_phasescreens}-elements array')
        if len(self.local_inputs['wind_direction'].value) != self.n_phasescreens:
            raise ValueError('Wind direction input must be a {self.n_phasescreens}-elements array')

        n = self.n_phasescreens
        self.delta_time_xp = self.xp.zeros((), dtype=self.dtype)
        self.scale_coef = self.xp.zeros(1, dtype=self.dtype)
        self.win_matrix = self.xp.zeros((len(self.layer_lists), n, 2, 3), dtype=self.dtype)
        self.rot_matrix = self.xp.zeros((n, 2, 3), dtype=self.dtype)
        # Only the x offset of the window matrices changes at each step
        self.win_matrix[..., :2] = self.xp.eye(2, dtype=self.dtype)
        self.last_effective_position = self.win_matrix[0, :, 1, 2]
        self.pixel_layer_xp = self.to_xp(self.pixel_layer, dtype=self.dtype)
        self.layer_center_xp = (self.pixel_layer_xp - 1) / 2
        self.screen_size_xp = self.to_xp(self.phasescreens_sizes_array, dtype=self.dtype)

        # Reallocated inputs raise an error, see BaseProcessingObj.check_input_ptrs()
        self.build_stream()

    def prepare_trigger(self, t):
        """Update the device time step, only when it changes (no transfers)."""
        super().prepare_trigger(t)
        delta_time = np.float64(self.t_to_seconds(self.current_time - self.last_t))
        if delta_time != self.delta_time:
            self.delta_time = delta_time
            self.delta_time_xp.fill(delta_time)

    def trigger_code(self):
        """Compute all layer lists from the inputs.

        Computes the seeing scale coefficient (zero for seeing <= 0) and the rotation
        matrices with fused kernels. Then, for each layer list, a fused kernel
        accumulates and cycles the positions (self.last_positions); the effective
        position, including the extra offset from self.extra_delta_times, is the x
        offset of the window matrices. Finally, for each layer: window extraction with
        linear interpolation along x (self.win_matrix), rotation (self.rot_matrix),
        both with lib.affine_transform.affine_transform(), and multiplication by
        self.scale_coef.

        On GPU, in addition to kernel launches, there are small temporary
        allocations inside cupyx.scipy.ndimage.affine_transform: in the CUDA graph,
        they are taken from its own memory pool (see BaseProcessingObj.capture_stream()).
        """
        wind_speed = self.local_inputs['wind_speed'].value
        wind_direction = self.local_inputs['wind_direction'].value
        _seeing_scale(self.scale_coef, self.local_inputs['seeing'].value,
                      self.seeing_scale_factor, xp=self.xp)
        _rotation_matrix(*[self.rot_matrix[:, i, j] for i in range(2) for j in range(3)],
                         wind_direction, self.layer_center_xp, xp=self.xp)

        for layer_list, win_matrix, extra_delta_time, last_position in zip(
                self.layer_lists, self.win_matrix, self.extra_delta_times, self.last_positions):
            _positions(last_position, win_matrix[:, 1, 2], wind_speed, self.delta_time_xp,
                       extra_delta_time, self.pixel_layer_xp, self.screen_size_xp,
                       self.pixel_pitch, self.cycle_screens, xp=self.xp)
            for p, window, layer, win, rot in zip(self.phasescreens, self.windows, layer_list,
                                                  win_matrix, self.rot_matrix):
                affine_transform(p, win, window, xp=self.xp)
                affine_transform(window, rot, layer.phaseInNm, xp=self.xp)
                layer.phaseInNm *= self.scale_coef

    def post_trigger(self):
        """Host-side state update: last_t and generation_time of all output layers."""
        super().post_trigger()
        self.last_t = self.current_time
        for layer_list in self.layer_lists:
            for layer in layer_list:
                layer.generation_time = self.current_time

