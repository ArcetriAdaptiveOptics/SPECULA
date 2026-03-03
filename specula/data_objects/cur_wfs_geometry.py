import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class CurWfsGeometry(BaseDataObj):
    """
    Curvature Wavefront Sensor Geometry object.
    Holds the 3D mask array for subaperture integration.
    """
    def __init__(self,
                 size_pixels: int = 0,
                 rings_config: list = None,
                 masks=None,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Initialize a CurWFSGeometry object.

        Parameters
        ----------
        size_pixels : int
            Size of the sensor array in pixels (side of the square).
        rings_config : list of dict
            List of dictionaries defining the rings, e.g.:
            [ {'inner': 0, 'outer': 0.2, 'segments': 1},
              {'inner': 0.2, 'outer': 0.5, 'segments': 6}, ... ]
            Radii are normalized to 1 at the edge of the pupil.
        masks : array, optional
            Pre-computed masks array (used when restoring from file).
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU).
        precision : int, optional
            Precision for computation (0 for double, 1 for single).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if masks is not None:
            # Path used by `restore()` or cloning
            self.n_subaps = masks.shape[0]
            self.size = masks.shape[1]
            self.masks = self.to_xp(masks, dtype=self.dtype)

        elif rings_config is not None and size_pixels > 0:
            # Creation from scratch
            self.size = int(size_pixels)
            self.n_subaps = sum([r['segments'] for r in rings_config])

            # We generate the masks on the CPU with standard NumPy for simplicity and
            # then move them to the target GPU/CPU
            cpu_masks = np.zeros((self.n_subaps, self.size, self.size), dtype=np.float32)

            # Create polar grids
            y, x = np.ogrid[-self.size/2 : self.size/2, -self.size/2 : self.size/2]
            r = np.sqrt(x**2 + y**2) / (self.size/2) # Normalized to 1
            theta = np.arctan2(y, x)

            idx = 0
            for ring in rings_config:
                ring_mask = (r >= ring['inner']) & (r < ring['outer'])
                d_theta = 2 * np.pi / ring['segments']

                for i in range(ring['segments']):
                    # Isolate the sector
                    theta_min = -np.pi + i * d_theta
                    theta_max = theta_min + d_theta
                    sector_mask = (theta >= theta_min) & (theta < theta_max)
                    cpu_masks[idx, :, :] = (ring_mask & sector_mask).astype(np.float32)
                    idx += 1

            # Copy the array to the correct device (Cupy if GPU, Numpy if CPU)
            self.masks = self.to_xp(cpu_masks, dtype=self.dtype)

        else:
            # Empty initialization (safety fallback)
            self.n_subaps = 0
            self.size = 0
            self.masks = self.xp.empty((0, 0, 0), dtype=self.dtype)

    def get_flattened_masks(self):
        """
        Returns a 2D matrix (N_subaps, N_pixels_total) perfect for 
        fast matrix multiplication during slope calculation.
        """
        return self.masks.reshape(self.n_subaps, -1)

    def get_value(self):
        """ Returns the 3D masks array. Required by some BaseDataObj mechanisms. """
        return self.masks

    def set_value(self, v):
        """ Sets the masks array without reallocating, if shapes match. """
        assert v.shape == self.masks.shape, \
            f"Error: input array shape {v.shape} does not match masks shape {self.masks.shape}"
        self.masks[:] = self.to_xp(v)

    # -------------------------------------------------------------------------
    # Standard FITS I/O methods for Specula
    # -------------------------------------------------------------------------

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NSUBAPS'] = self.n_subaps
        hdr['SIZEPIX'] = self.size
        return hdr

    def save(self, filename, overwrite=False):
        """ Saves the object to a FITS file. """
        hdr = self.get_fits_header()
        # cpuArray fa il pull da CuPy a NumPy automaticamente se necessario
        fits.writeto(filename, cpuArray(self.masks), hdr, overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        """ Creates an empty object based on header info. """
        version = hdr.get('VERSION')
        if version != 1:
            raise ValueError(f"Unknown version {version} in header")

        n_subaps = hdr['NSUBAPS']
        size_pix = hdr['SIZEPIX']

        # Create empty object with correct dimensions
        # We pass a zero-filled array to force correct shape initialization
        import numpy as np
        empty_masks = np.zeros((n_subaps, size_pix, size_pix), dtype=np.float32)

        return CurWfsGeometry(masks=empty_masks,
                                       target_device_idx=target_device_idx)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """ Restores the object from a FITS file. """
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")

            masks = hdul[0].data
            size_pixels = hdr.get('SIZEPIX', masks.shape[1])

        return CurWfsGeometry(size_pixels=size_pixels,
                            masks=masks,
                            target_device_idx=target_device_idx)
