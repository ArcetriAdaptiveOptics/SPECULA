from specula.processing_objects.slopec import Slopec
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes

class CurWfsSlopec(Slopec):
    """
    Slope Computer for Curvature Wavefront Sensor.
    Computes the normalized difference between intra-focal and extra-focal fluxes.
    """
    def __init__(self,
                 cwfs_geometry, # The CurWFSGeometry object
                 sn: Slopes=None,
                 interleave: bool=False,
                 target_device_idx: int=None,
                 precision: int=None):

        # Geometry must be set BEFORE calling super().__init__
        # because the base class calls self.nslopes() which uses self.geometry.
        self.geometry = cwfs_geometry

        super().__init__(sn=sn, interleave=interleave,
                         target_device_idx=target_device_idx, precision=precision)

        # Modify Inputs: CWFS requires two images (Intra/Extra focal)
        if 'in_pixels' in self.inputs:
            del self.inputs['in_pixels']

        self.inputs['in_pixels1'] = InputValue(type=Pixels)
        self.inputs['in_pixels2'] = InputValue(type=Pixels)

        # Pre-allocate variables
        self.mask_matrix = None
        self._flux1 = None
        self._flux2 = None


    def nsubaps(self):
        return self.geometry.n_subaps


    def nslopes(self):
        # 1 signal per subaperture (curvature)
        return self.geometry.n_subaps


    def setup(self):
        super().setup()

        # 1. Get the flattened mask matrix from the geometry object
        self.mask_matrix = self.geometry.get_flattened_masks()

        # 2. Allocate flux vectors
        self._flux1 = self.xp.zeros(self.nsubaps(), dtype=self.dtype)
        self._flux2 = self.xp.zeros(self.nsubaps(), dtype=self.dtype)


    def trigger_code(self):
        # 1. Flatten input images
        p1 = self.local_inputs['in_pixels1'].pixels.ravel()
        p2 = self.local_inputs['in_pixels2'].pixels.ravel()

        # 2. Integrate flux using Matrix Multiplication
        # (N_sectors, N_pix) @ (N_pix) -> (N_sectors)
        self._flux1[:] = self.mask_matrix @ p1
        self._flux2[:] = self.mask_matrix @ p2

        # 3. Compute Curvature Signal
        sum_flux = self._flux1 + self._flux2

        # Avoid division by zero
        sum_flux = self.xp.where(sum_flux < 1e-9, 1.0, sum_flux)

        signal = (self._flux1 - self._flux2) / sum_flux

        # 4. Store results
        self.slopes.slopes[:] = signal

        # Update telemetry
        self.flux_per_subaperture_vector.value[:] = sum_flux
        self.total_counts.value[0] = self.xp.sum(sum_flux)
        self.subap_counts.value[0] = self.xp.mean(sum_flux)
