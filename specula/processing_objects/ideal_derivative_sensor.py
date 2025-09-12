from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.data_objects.simul_params import SimulParams
from specula.lib.extrapolation_2d import calculate_extrapolation_indices_coeffs, apply_extrapolation
from specula import cpuArray, RAD2ASEC


class IdealDerivativeSensor(BaseProcessingObj):
    """
    Ideal derivative sensor that computes slopes from wavefront derivatives.
    
    This sensor extrapolates the phase outside the pupil mask using linear extrapolation,
    then computes X and Y derivatives to generate slopes for each subaperture.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 subapdata: SubapData,
                 fov: float,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Initialize the ideal derivative sensor.
        
        Args:
            subapdata: Subaperture data object defining the geometry
            pixel_pitch: Pixel pitch in meters
            fov: Field of view in arcseconds (radius). This is used to get the same scale factor of a SH sensor
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.subapdata = subapdata
        self.simul_params = simul_params
        self.pixel_pitch = self.simul_params.pixel_pitch
        if fov <= 0:
            raise ValueError("Field of view must be positive.")
        self.fov = fov

        # Conversion factor from derivative to slopes
        # slope_value = derivative [nm] * 1e-9 / pixel_pitch [m] * rad2asec / FoV [asec] (radius)
        self.slope_factor = 1e-9 * RAD2ASEC / (self.pixel_pitch * self.fov / 2.0)

        # Initialize slopes output
        self.slopes = Slopes(length=self.subapdata.n_subaps * 2,
                           interleave=False,
                           target_device_idx=target_device_idx,
                           precision=precision)
        self.slopes.single_mask = self.subapdata.single_mask()
        self.slopes.display_map = self.subapdata.display_map

        # Cache for extrapolation
        self._edge_pixels = None
        self._reference_indices = None
        self._coefficients = None
        self._valid_indices = None
        self._subap_indices = None

        n_subaps = self.subapdata.n_subaps
        self.sx = self.xp.zeros(n_subaps, dtype=self.dtype)
        self.sy = self.xp.zeros(n_subaps, dtype=self.dtype)

        # Setup inputs and outputs
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_slopes'] = self.slopes

    def setup(self):
        """Setup the sensor geometry and caching."""
        super().setup()

        # Get input electric field for initialization
        in_ef = self.local_inputs['in_ef']

        # Pre-compute subaperture indices for efficiency
        self._compute_subap_indices(in_ef.size)

    def _compute_subap_indices(self, ef_size):
        """Pre-compute indices for each subaperture."""
        npixels = ef_size[0]
        nx = self.subapdata.nx
        np_sub = npixels // nx  # pixels per subaperture

        self._subap_indices = []

        for i in range(self.subapdata.n_subaps):
            subap_map = self.subapdata.display_map[i]
            j = subap_map % nx  # column
            i_row = subap_map // nx  # row

            # Center coordinates of this subaperture
            x_center = np_sub / 2.0 + np_sub * j
            y_center = np_sub / 2.0 + np_sub * i_row

            # Create mask for this subaperture
            x_start = int(self.xp.round(x_center - np_sub / 2.0))
            x_end = int(self.xp.round(x_center + np_sub / 2.0))
            y_start = int(self.xp.round(y_center - np_sub / 2.0))
            y_end = int(self.xp.round(y_center + np_sub / 2.0))

            # Create indices for this subaperture
            y_indices, x_indices = self.xp.meshgrid(
                self.xp.arange(y_start, y_end),
                self.xp.arange(x_start, x_end),
                indexing='ij'
            )

            # Flatten and store
            subap_idx = self.xp.ravel_multi_index(
                (y_indices.ravel(), x_indices.ravel()),
                ef_size
            )
            self._subap_indices.append(subap_idx)

    def trigger_code(self):
        """Main processing: extrapolate phase and compute slopes."""
        in_ef = self.local_inputs['in_ef']
        n_subaps = self.subapdata.n_subaps

        # Step 1: Extrapolate phase outside the pupil
        phase_extrapolated = apply_extrapolation(
            in_ef.phaseInNm,
            self._edge_pixels,
            self._reference_indices,
            self._coefficients,
            self._valid_indices,
            xp=self.xp
        )

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.imshow(phase_extrapolated, cmap='jet', interpolation='nearest')
            plt.colorbar(label='Phase [nm]')
            plt.title('Extrapolated Phase')

        # Step 2: Compute derivatives
        dx_phase = self.xp.gradient(phase_extrapolated, axis=1)
        dy_phase = self.xp.gradient(phase_extrapolated, axis=0)

        # Step 3: Mask derivatives to valid regions only
        mask = in_ef.A > 0.5
        dx_phase = dx_phase * mask
        dy_phase = dy_phase * mask

        if plot_debug:
            plt.figure(figsize=(10, 5))
            plt.imshow(mask, cmap='gray', interpolation='nearest')
            plt.title('Valid Mask')
            plt.colorbar(label='Mask Value')

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(dx_phase, cmap='jet', interpolation='nearest')
            plt.colorbar(label='X Derivative [nm]')
            plt.title('X Derivative')
            plt.subplot(1, 2, 2)
            plt.imshow(dy_phase, cmap='jet', interpolation='nearest')
            plt.colorbar(label='Y Derivative [nm]')
            plt.title('Y Derivative')

        # Step 4: Vectorized slope computation using mean
        dx_flat = dx_phase.ravel()
        dy_flat = dy_phase.ravel()

        # Stack all subaperture indices for vectorized extraction
        max_subap_size = max(len(idx) for idx in self._subap_indices)
        subap_indices_matrix = self.xp.full((n_subaps, max_subap_size), -1, dtype=int)
        subap_masks = self.xp.zeros((n_subaps, max_subap_size), dtype=bool)

        for i, idx in enumerate(self._subap_indices):
            subap_indices_matrix[i, :len(idx)] = idx
            subap_masks[i, :len(idx)] = True

        # Extract all derivatives at once
        dx_all = dx_flat[subap_indices_matrix]  # Shape: (n_subaps, max_subap_size)
        dy_all = dy_flat[subap_indices_matrix]

        # Apply masks to handle different subaperture sizes
        dx_all = dx_all * subap_masks
        dy_all = dy_all * subap_masks

        # Find valid pixels (non-zero derivatives) for all subapertures
        valid_masks = ((self.xp.abs(dx_all) + self.xp.abs(dy_all)) > 0) & subap_masks

        # Vectorized mean computation
        # Set invalid pixels to 0 for mean calculation
        dx_masked = dx_all * valid_masks
        dy_masked = dy_all * valid_masks

        if plot_debug:
            plt.figure(figsize=(10, 5))
            plt.imshow(subap_masks, cmap='gray', interpolation='nearest')
            plt.title('Valid Masks')
            plt.colorbar(label='Mask Value')

        # Count valid pixels per subaperture
        valid_counts = self.xp.sum(valid_masks, axis=1)

        # Compute sums and divide by counts (avoiding division by zero)
        # This is a vectorized operation equivalent to the average in the subapertures
        dx_sums = self.xp.sum(dx_masked, axis=1)
        dy_sums = self.xp.sum(dy_masked, axis=1)

        # Avoid division by zero
        valid_counts = self.xp.where(valid_counts > 0, valid_counts, 1)

        self.sx[:] = (dx_sums / valid_counts) * self.slope_factor
        self.sy[:] = (dy_sums / valid_counts) * self.slope_factor

        # Set slopes to 0 where no valid pixels
        no_valid = self.xp.sum(valid_masks, axis=1) == 0
        self.sx[no_valid] = 0.0
        self.sy[no_valid] = 0.0

        if plot_debug:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(self.sx.reshape((self.subapdata.nx, self.subapdata.ny)), cmap='jet', interpolation='nearest')
            plt.colorbar(label='X Slope [nm]')
            plt.title('X Slope')

            plt.subplot(1, 2, 2)
            plt.imshow(self.sy.reshape((self.subapdata.nx, self.subapdata.ny)), cmap='jet', interpolation='nearest')
            plt.colorbar(label='Y Slope [nm]')
            plt.title('Y Slope')

            plt.show()

    def post_trigger(self):
        super().post_trigger()
        # Store slopes
        self.slopes.xslopes = self.sx
        self.slopes.yslopes = self.sy
        # Update generation time
        self.slopes.generation_time = self.current_time

    def prepare_trigger(self, t):
        """Prepare for trigger execution."""
        super().prepare_trigger(t)

        # Setup extrapolation if not already done
        if self._edge_pixels is None:
            in_ef = self.local_inputs['in_ef']

            good_pixels_mask = cpuArray(in_ef.A) > 0.5

            # Calculate extrapolation indices and coefficients
            (self._edge_pixels,
            self._reference_indices,
            self._coefficients,
            self._valid_indices) = calculate_extrapolation_indices_coeffs(good_pixels_mask)

            # Convert to target device
            self._edge_pixels = self.to_xp(self._edge_pixels)
            self._reference_indices = self.to_xp(self._reference_indices)
            self._coefficients = self.to_xp(self._coefficients)
            self._valid_indices = self.to_xp(self._valid_indices)
