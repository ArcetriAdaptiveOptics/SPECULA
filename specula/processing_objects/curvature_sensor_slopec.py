from specula.processing_objects.slopec import Slopec
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels

class CurvatureSensorSlopec(Slopec):
    def __init__(self,
                 cwfs_geometry, # A custom object that contains the segment map
                 **kwargs):

        # Save the geometry (which contains n_subaps)
        self.geometry = cwfs_geometry

        super().__init__(**kwargs)

        # CWFS needs 2 inputs (I1 and I2)
        # Rename the base inputs
        del self.inputs['in_pixels']
        self.inputs['in_pixels1'] = InputValue(type=Pixels)
        self.inputs['in_pixels2'] = InputValue(type=Pixels)

        self.mask_matrix = None  # To be allocated in setup

        # Pre-allocate vectors for the fluxes
        self._flux1 = None
        self._flux2 = None

    def nsubaps(self):
        return self.geometry.n_subaps

    def nslopes(self):
        # In CWFS, each subaperture has 1 measurement (curvature), not 2 (X,Y)
        return self.geometry.n_subaps

    def setup(self):
        super().setup()
        # Allocate the sparse or 3D mask matrix on GPU
        # shape: (n_subaps, total_N_pixels)
        self.mask_matrix = self.to_xp(self.geometry.get_flattened_masks())
        self._flux1 = self.xp.zeros(self.nsubaps(), dtype=self.dtype)
        self._flux2 = self.xp.zeros(self.nsubaps(), dtype=self.dtype)

    def trigger_code(self):
        # No FOR loops here! Fully vectorized for the GPU.

        # 1. Retrieve the flat (1D) images
        p1 = self.local_inputs['in_pixels1'].pixels.ravel()
        p2 = self.local_inputs['in_pixels2'].pixels.ravel()

        # 2. Integrate the flux in the sectors via matrix multiplication
        # mask_matrix is [n_subaps, n_pixels], p1 is [n_pixels]
        # The result is a vector [n_subaps]
        self._flux1[:] = self.mask_matrix @ p1
        self._flux2[:] = self.mask_matrix @ p2

        # 3. Calculate the signal S = (I1 - I2) / (I1 + I2)
        sum_flux = self._flux1 + self._flux2

        # Avoid division by zero
        sum_flux = self.xp.where(sum_flux < 1e-6, 1.0, sum_flux)

        signal = (self._flux1 - self._flux2) / sum_flux

        # 4. Write to the Slopes object (used here to store curvature)
        self.slopes.slopes[:] = signal

        # Update the flux metric for telemetry
        self.flux_per_subaperture_vector.value[:] = sum_flux
