import numpy as np
from typing import Optional, Union, Tuple, List

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula.connections import InputValue

class ExtendedSource(BaseProcessingObj):
    """
    ExtendedSource class to compute extended sources (list of 3D points) for pyramid wavefront sensing.

    Args:
        simul_params (SimulParams): Simulation parameters.
        wavelengthInNm (float): Wavelength in nanometers.
        source_type (str): Type of source ('POINT_SOURCE', 'TOPHAT', 'GAUSS', 'FROM_PSF').
        sampling_lambda_over_d (float): Sampling factor in units of λ/D. Larger values mean less points.
        size_obj (Optional[float]): Size of the object in arcseconds. Required for 'TOPHAT' and 'GAUSS' sources.
        sampling_type (str): Sampling type ('CARTESIAN', 'POLAR', 'RINGS').
        layer_height (Optional[List[float]]): Heights of layers in meters. Used for 3D sources (sodium beacon).
        intensity_profile (Optional[List[float]]): Intensity profile for each layer. Used for 3D sources (sodium beacon).
        focus_height (Optional[float]): Height of the focus in meters. Used for 3D sources (sodium beacon).
        tt_profile (Optional[np.ndarray]): Tip/tilt profile for each layer. Used for 3D sources (sodium beacon).
        n_rings (Optional[int]): Number of rings for 'RINGS' sampling. Default is 0.
        flux_threshold (float): Threshold for flux. Points with flux below this value are discarded.
        initial_psf (Optional[np.ndarray]): PSF array for 'FROM_PSF' source type to be used for initialization.
        pixel_scale_psf (Optional[float]): Pixel scale of the PSF in arcseconds. Required for 'FROM_PSF' source type.
        target_device_idx (int): Index of the target device for computation. 0 is first GPU, -1 is CPU.
        precision (int): Precision for computation (e.g., 32 or 64 bits). 1 is single precision, 0 is double precision.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 source_type: str,                      # 'POINT_SOURCE', 'TOPHAT', 'GAUSS', 'FROM_PSF'
                 sampling_lambda_over_d: float = 1.0,   # Sampling factor in units of λ/D
                 size_obj: Optional[float] = None,      # size in arcsec
                 sampling_type: str = 'CARTESIAN',      # 'CARTESIAN', 'POLAR', 'RINGS'
                 layer_height: Optional[List[float]] = None,    
                 intensity_profile: Optional[List[float]] = None,
                 focus_height: Optional[float] = None,
                 tt_profile: Optional[np.ndarray] = None,
                 n_rings: Optional[int] = None,
                 flux_threshold: float = 0.0,
                 initial_psf: Optional[np.ndarray] = None,
                 pixel_scale_psf: Optional[float] = None,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Store parameters
        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.zenithAngleInDeg = self.simul_params.zenithAngleInDeg
        self.airmass = 1. / np.cos(np.radians(self.simul_params.zenithAngleInDeg), dtype=self.dtype)
        
        self.wavelengthInNm = wavelengthInNm
        self.sampling_lambda_over_d = sampling_lambda_over_d
        self.d_tel = self.pixel_pupil * self.pixel_pitch
        self.source_type = source_type
        self.size_obj = size_obj
        self.sampling_type = sampling_type
        if layer_height is not None:
            layer_height = [h * self.airmass for h in layer_height]
        self.layer_height = layer_height or []
        self.intensity_profile = intensity_profile or []
        if focus_height is not None:
            focus_height = focus_height * self.airmass
        else:
            focus_height = np.inf
        self.focus_height = focus_height
        self.tt_profile = tt_profile
        self.n_rings = n_rings or 0
        self.flux_threshold = flux_threshold
        self.psf = BaseValue()
        if initial_psf is not None:
            self.psf.value = self.to_xp(initial_psf, dtype=self.dtype)
        else:
            self.psf.value = self.xp.zeros((3, 3), dtype=self.dtype)
            self.psf.value[1, 1] = 1.0  # Default initial PSF is a delta function
        self.pixel_scale_psf = pixel_scale_psf

        # Validate parameters
        self._validate_parameters()

        # Determine if 3D
        self.is_3d = self._check_if_3d()

        # Output arrays
        self.npoints = 0
        self.coeff_tiltx = None
        self.coeff_tilty = None
        self.coeff_focus = None
        self.coeff_flux = None
        self.xx_arcsec = None
        self.yy_arcsec = None

        # Add input for PSF updates
        self.inputs['psf'] = InputValue(type=BaseValue, optional=True)

        # Outputs
        self.outputs['coeff'] = BaseValue(target_device_idx=self.target_device_idx)

        # Compute coefficients
        self.compute()

    def _validate_parameters(self):
        """Validate input parameters"""
        if self.source_type not in ['POINT_SOURCE', 'TOPHAT', 'GAUSS', 'FROM_PSF']:
            raise ValueError(f"Invalid source_type: {self.source_type}")

        if self.sampling_type not in ['CARTESIAN', 'POLAR', 'RINGS']:
            raise ValueError(f"Invalid sampling_type: {self.sampling_type}")

        if self.source_type in ['TOPHAT', 'GAUSS'] and self.size_obj is None:
            raise ValueError(f"{self.source_type} requires size_obj parameter")

        if self.source_type == 'FROM_PSF':
            if self.psf.value is None:
                raise ValueError("FROM_PSF requires psf parameter")
            if self.pixel_scale_psf is None:
                raise ValueError("FROM_PSF requires pixel_scale_psf parameter")

    def _check_if_3d(self) -> bool:
        """Check if this is a 3D extended source"""
        if len(self.layer_height) == 1 and self.focus_height is not None:
            return self.layer_height[0] != self.focus_height
        return len(self.layer_height) > 1

    def compute(self):
        """Main computation method"""
        if self.is_3d:
            result = self._compute_3d()
        else:
            result = self._compute_2d()

        # Store results
        self.xx_arcsec = result['xx_arcsec']
        self.yy_arcsec = result['yy_arcsec']
        self.coeff_tiltx = self.to_xp(result['coeff_tiltx'])
        self.coeff_tilty = self.to_xp(result['coeff_tilty'])
        self.coeff_focus = self.to_xp(result['coeff_focus'])
        self.coeff_flux = self.to_xp(result['coeff_flux'])
        self.npoints = len(self.coeff_tiltx)

        # Apply flux threshold if needed
        if self.flux_threshold > 0:
            self._apply_flux_threshold()

        # Update outputs
        self.outputs['coeff'].value = self.xp.column_stack([
                                                    self.coeff_tiltx,
                                                    self.coeff_tilty,
                                                    self.coeff_focus,
                                                    self.coeff_flux
                                                ])

    def _compute_2d(self) -> dict:
        """Compute 2D extended source"""
        # Object sampling in arcsec
        sec2rad = 4.848e-6
        obj_sampling = self.sampling_lambda_over_d * (self.wavelengthInNm/1e9) / self.d_tel / sec2rad

        if self.source_type == 'POINT_SOURCE':
            return self._compute_point_source()
        elif self.source_type == 'TOPHAT':
            return self._compute_tophat(obj_sampling)
        elif self.source_type == 'GAUSS':
            return self._compute_gauss(obj_sampling)
        elif self.source_type == 'FROM_PSF':
            return self._compute_from_psf(obj_sampling)
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")

    def _compute_point_source(self) -> dict:
        """Compute point source"""
        return {
            'xx_arcsec': np.array([0.0]),
            'yy_arcsec': np.array([0.0]),
            'coeff_tiltx': np.array([0.0]),
            'coeff_tilty': np.array([0.0]),
            'coeff_focus': np.array([0.0]),
            'coeff_flux': np.array([1.0])
        }

    def _compute_tophat(self, obj_sampling: float) -> dict:
        """Compute TOPHAT extended source"""
        if self.size_obj is None:
            raise ValueError("TOPHAT source requires size_obj parameter")

        if self.sampling_type == 'CARTESIAN':
            return self._compute_tophat_cartesian(obj_sampling)
        elif self.sampling_type == 'POLAR':
            return self._compute_tophat_polar(obj_sampling)
        elif self.sampling_type == 'RINGS':
            return self._compute_tophat_rings(obj_sampling)
        else:
            raise ValueError(f"Unknown sampling type: {self.sampling_type}")

    def _compute_tophat_cartesian(self, obj_sampling: float) -> dict:
        """Compute TOPHAT with Cartesian sampling"""
        n_point_diam = int(np.round(self.size_obj / obj_sampling / 2)) * 2
        half_size = self.size_obj / 2.0

        # Create coordinate grid
        coords = np.linspace(-half_size, half_size, n_point_diam)
        xx, yy = np.meshgrid(coords, coords)
        xx = xx.flatten()
        yy = yy.flatten()

        # Keep only points inside circle
        rr = np.sqrt(xx**2 + yy**2)
        valid_mask = rr <= half_size

        xx_arcsec = xx[valid_mask]
        yy_arcsec = yy[valid_mask]
        n_points = len(xx_arcsec)

        # Convert to tip/tilt coefficients
        coeff_tiltx = self._angle_to_tip(xx_arcsec)
        coeff_tilty = self._angle_to_tip(yy_arcsec)
        coeff_focus = np.zeros(n_points)
        coeff_flux = np.ones(n_points) / n_points

        return {
            'xx_arcsec': xx_arcsec,
            'yy_arcsec': yy_arcsec,
            'coeff_tiltx': coeff_tiltx,
            'coeff_tilty': coeff_tilty,
            'coeff_focus': coeff_focus,
            'coeff_flux': coeff_flux
        }

    def _compute_tophat_polar(self, obj_sampling: float) -> dict:
        """Compute TOPHAT with polar sampling"""
        if self.size_obj is None:
            raise ValueError("TOPHAT source requires size_obj parameter")

        # Number of radial points along radius of extended object
        n_radial_points = int(np.round(self.size_obj / obj_sampling / 2))
        rr_arcsec = np.linspace(0, self.size_obj/2, n_radial_points)

        # Central Point
        xx_arcsec = [0.0]
        yy_arcsec = [0.0]

        # Points in concentric rings
        for jj in range(1, n_radial_points):
            perim = 2 * np.pi * rr_arcsec[jj]
            n_ring_points = int(np.round(perim / obj_sampling))

            if n_ring_points > 0:
                # Create angles for this ring, excluding redundant point
                theta_ring = np.linspace(0, 2*np.pi, n_ring_points + 1)[:-1]

                # Convert to cartesian coordinates
                xx_ring = rr_arcsec[jj] * np.cos(theta_ring)
                yy_ring = rr_arcsec[jj] * np.sin(theta_ring)

                # Add to total arrays
                xx_arcsec.extend(xx_ring)
                yy_arcsec.extend(yy_ring)

        # Convert to numpy arrays
        xx_arcsec = np.array(xx_arcsec)
        yy_arcsec = np.array(yy_arcsec)
        n_points = len(xx_arcsec)

        # Convert to tip/tilt coefficients
        coeff_tiltx = self._angle_to_tip(xx_arcsec)
        coeff_tilty = self._angle_to_tip(yy_arcsec)
        coeff_focus = np.zeros(n_points)
        coeff_flux = np.ones(n_points) / n_points

        return {
            'xx_arcsec': xx_arcsec,
            'yy_arcsec': yy_arcsec,
            'coeff_tiltx': coeff_tiltx,
            'coeff_tilty': coeff_tilty,
            'coeff_focus': coeff_focus,
            'coeff_flux': coeff_flux
        }

    def _compute_tophat_rings(self, obj_sampling: float) -> dict:
        """Compute TOPHAT with rings sampling"""
        if self.size_obj is None:
            raise ValueError("TOPHAT source requires size_obj parameter")

        # Number of rings
        if self.n_rings > 0:
            n_rings = self.n_rings
        else:
            # Default: based on diffraction-limited resolution
            n_rings = int(np.round(self.size_obj/2 / (5 * (self.wavelengthInNm/1e9) / self.d_tel / 4.848e-6)))

        # Ring geometry
        size_ring = (self.size_obj/2) / n_rings
        radius_rings = size_ring/2 + np.arange(n_rings) * size_ring

        # Number of points per ring (based on circumference)
        np_rings = np.ceil(2 * np.pi * radius_rings / obj_sampling).astype(int)

        # Total number of points (rings + central point)
        total_points = int(np.sum(np_rings)) + 1

        # Initialize arrays
        xx_arcsec = [0.0]  # Central point
        yy_arcsec = [0.0]

        # Generate points for each ring
        for j in range(n_rings):
            if np_rings[j] > 0:
                # Angles for this ring
                angles = 2 * np.pi / np_rings[j] * np.arange(np_rings[j])

                # Convert to cartesian coordinates
                xx_ring = radius_rings[j] * np.cos(angles)
                yy_ring = radius_rings[j] * np.sin(angles)

                # Add to arrays
                xx_arcsec.extend(xx_ring)
                yy_arcsec.extend(yy_ring)

        # Convert to numpy arrays
        xx_arcsec = np.array(xx_arcsec)
        yy_arcsec = np.array(yy_arcsec)
        n_points = len(xx_arcsec)

        # Convert to tip/tilt coefficients
        coeff_tiltx = self._angle_to_tip(xx_arcsec)
        coeff_tilty = self._angle_to_tip(yy_arcsec)
        coeff_focus = np.zeros(n_points)
        coeff_flux = np.ones(n_points) / n_points  # Equal flux per point

        return {
            'xx_arcsec': xx_arcsec,
            'yy_arcsec': yy_arcsec,
            'coeff_tiltx': coeff_tiltx,
            'coeff_tilty': coeff_tilty,
            'coeff_focus': coeff_focus,
            'coeff_flux': coeff_flux
        }

    def _compute_gauss(self, obj_sampling: float) -> dict:
        """Compute Gaussian extended source"""
        if self.size_obj is None:
            raise ValueError("GAUSS source requires size_obj parameter (FWHM)")

        if self.sampling_type == 'CARTESIAN':
            return self._compute_gauss_cartesian(obj_sampling)
        elif self.sampling_type == 'RINGS':
            return self._compute_gauss_rings(obj_sampling)
        else:
            raise ValueError(f"GAUSS sampling type {self.sampling_type} not implemented")

    def _compute_gauss_cartesian(self, obj_sampling: float) -> dict:
        """Compute Gaussian with Cartesian sampling"""
        # Convert FWHM to sigma
        sigma_arcsec = self.size_obj / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_pix = sigma_arcsec / obj_sampling

        # Compute up to 3 sigma
        max_extent = 3.0 * sigma_arcsec
        max_pix = max_extent / obj_sampling

        # Create grid
        dim_tab = int(np.round(2.0 * max_pix / 2)) * 2
        coords = (np.arange(dim_tab) - dim_tab/2 + 0.5) * obj_sampling
        xx, yy = np.meshgrid(coords, coords)

        # Compute Gaussian
        gaussian = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_arcsec**2))

        # Select points that contribute significantly
        valid_mask = gaussian >= 0.1 * np.max(gaussian)

        xx_arcsec = xx[valid_mask]
        yy_arcsec = yy[valid_mask]
        flux_weights = gaussian[valid_mask]
        flux_weights /= np.sum(flux_weights)  # Normalize

        coeff_tiltx = self._angle_to_tip(xx_arcsec)
        coeff_tilty = self._angle_to_tip(yy_arcsec)
        coeff_focus = np.zeros(len(xx_arcsec))

        return {
            'xx_arcsec': xx_arcsec,
            'yy_arcsec': yy_arcsec,
            'coeff_tiltx': coeff_tiltx,
            'coeff_tilty': coeff_tilty,
            'coeff_focus': coeff_focus,
            'coeff_flux': flux_weights
        }

    def _compute_gauss_rings(self, obj_sampling: float) -> dict:
        """Compute Gaussian with rings sampling"""
        if self.size_obj is None:
            raise ValueError("GAUSS source requires size_obj parameter (FWHM)")

        # Convert FWHM to sigma
        sigma_arcsec = self.size_obj / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Compute up to 3 sigma
        max_extent = 3.0 * sigma_arcsec

        # Number of rings
        if self.n_rings > 0:
            n_rings = self.n_rings
        else:
            # Default: based on diffraction-limited resolution
            n_rings = int(np.round(max_extent / (5 * (self.wavelengthInNm/1e9) / self.d_tel / 4.848e-6)))

        # Ring geometry
        size_ring = max_extent / n_rings
        radius_rings = size_ring/2 + np.arange(n_rings) * size_ring

        # Gaussian intensity profile at each ring radius
        gauss_prof = np.exp(-radius_rings**2 / (2 * sigma_arcsec**2))

        # Number of points per ring (based on circumference)
        np_rings = np.ceil(2 * np.pi * radius_rings / obj_sampling).astype(int)

        # Total number of points (rings + central point)
        total_points = int(np.sum(np_rings)) + 1

        # Initialize arrays
        xx_arcsec = [0.0]  # Central point
        yy_arcsec = [0.0]
        flux_percent = [1.0]  # Central point has exp(0) = 1

        # Generate points for each ring
        for j in range(n_rings):
            if np_rings[j] > 0:
                # Angles for this ring
                angles = 2 * np.pi / np_rings[j] * np.arange(np_rings[j])

                # Convert to cartesian coordinates
                xx_ring = radius_rings[j] * np.cos(angles)
                yy_ring = radius_rings[j] * np.sin(angles)

                # Add to arrays
                xx_arcsec.extend(xx_ring)
                yy_arcsec.extend(yy_ring)

                # All points in this ring have the same Gaussian intensity
                flux_percent.extend([gauss_prof[j]] * np_rings[j])

        # Convert to numpy arrays
        xx_arcsec = np.array(xx_arcsec)
        yy_arcsec = np.array(yy_arcsec)
        flux_percent = np.array(flux_percent)

        # Normalize flux
        flux_percent = flux_percent / np.sum(flux_percent)

        # Convert to tip/tilt coefficients
        coeff_tiltx = self._angle_to_tip(xx_arcsec)
        coeff_tilty = self._angle_to_tip(yy_arcsec)
        coeff_focus = np.zeros(len(xx_arcsec))

        return {
            'xx_arcsec': xx_arcsec,
            'yy_arcsec': yy_arcsec,
            'coeff_tiltx': coeff_tiltx,
            'coeff_tilty': coeff_tilty,
            'coeff_focus': coeff_focus,
            'coeff_flux': flux_percent
        }

    def _compute_from_psf(self, obj_sampling: float) -> dict:
        """Compute extended source from PSF"""

        psf = self.psf.value
        sPSF = psf.shape
        sPSFarcsec = np.array(sPSF) * self.pixel_scale_psf

        extobj_maxpix = int(np.round(sPSFarcsec[0] / obj_sampling))
        extobj_maxarcsec = extobj_maxpix * obj_sampling

        if self.sampling_type == 'CARTESIAN':
            # Make odd if even for proper centering
            if extobj_maxpix % 2 == 0:
                extobj_maxpix += 1

            # Create coordinate grids
            half_size = extobj_maxarcsec / 2
            coords = np.linspace(-half_size, half_size, extobj_maxpix)
            xx, yy = np.meshgrid(coords, coords)

            # Flatten for interpolation
            xx_arcsec = xx.flatten()
            yy_arcsec = yy.flatten()

            # Convert to PSF pixel coordinates for interpolation
            xx_psf = xx_arcsec / self.pixel_scale_psf + sPSF[0]/2
            yy_psf = yy_arcsec / self.pixel_scale_psf + sPSF[0]/2

            # Interpolate PSF values
            from scipy.interpolate import RectBivariateSpline
    
            # Create interpolation function
            x_psf = np.arange(sPSF[1])
            y_psf = np.arange(sPSF[0])
            interp_func = RectBivariateSpline(y_psf, x_psf, cpuArray(psf), kx=1, ky=1)

            # Interpolate at desired points (with boundary check)
            flux_percent = []
            valid_xx = []
            valid_yy = []

            for i in range(len(xx_arcsec)):
                x_coord = xx_psf[i]
                y_coord = yy_psf[i]

                # Check if coordinates are within PSF bounds
                if (0 <= x_coord < sPSF[1]-1) and (0 <= y_coord < sPSF[0]-1):
                    flux_val = float(interp_func(y_coord, x_coord)[0, 0])
                    flux_percent.append(flux_val)
                    valid_xx.append(xx_arcsec[i])
                    valid_yy.append(yy_arcsec[i])

            xx_arcsec = np.array(valid_xx)
            yy_arcsec = np.array(valid_yy)
            flux_percent = np.array(flux_percent)

        elif self.sampling_type == 'POLAR':
            # Polar sampling similar to IDL version
            rPol = np.arange(int(np.round((extobj_maxpix + 1) / 2)))
            naPol = rPol * 6
            naPol[0] = 1  # Central point

            # Handle sampling distance step if specified
            if hasattr(self, 'sampl_dist_step') and self.sampl_dist_step > 1:
                idxSDS = np.where(rPol % self.sampl_dist_step == 0)[0]
                idxGood = []

            npointTot = int(np.sum(naPol))
            pol_coords = np.zeros((2, npointTot))

            kaPol = 0
            for iaPol in range(len(naPol)):
                for jaPol in range(int(naPol[iaPol])):
                    pol_coords[0, kaPol] = 360.0 / naPol[iaPol] * jaPol  # angle
                    pol_coords[1, kaPol] = iaPol * sPSF[0] / (2 * len(naPol) - 1)  # radius

                    if hasattr(self, 'sampl_dist_step') and self.sampl_dist_step > 1:
                        if np.min(np.abs(iaPol - idxSDS)) == 0:
                            idxGood.append(kaPol)

                    kaPol += 1

            # Convert polar to rectangular coordinates
            angles_rad = np.deg2rad(pol_coords[0, :])
            radii = pol_coords[1, :]

            xxInterpol = radii * np.cos(angles_rad)
            yyInterpol = radii * np.sin(angles_rad)

            # Convert to arcsec
            xx_arcsec = xxInterpol * self.pixel_scale_psf
            yy_arcsec = yyInterpol * self.pixel_scale_psf

            # Adjust for PSF indexing
            xxInterpol += sPSF[0] / 2
            yyInterpol += sPSF[0] / 2

            # Apply sampling distance step filtering if needed
            if hasattr(self, 'sampl_dist_step') and self.sampl_dist_step > 1:
                xx_arcsec = xx_arcsec[idxGood]
                yy_arcsec = yy_arcsec[idxGood]
                xxInterpol = xxInterpol[idxGood]
                yyInterpol = yyInterpol[idxGood]

            # Interpolate PSF values
            from scipy.interpolate import RectBivariateSpline
            x_psf = np.arange(sPSF[1])
            y_psf = np.arange(sPSF[0])
            interp_func = RectBivariateSpline(y_psf, x_psf, psf, kx=1, ky=1)

            flux_percent = []
            for i in range(len(xx_arcsec)):
                x_coord = xxInterpol[i]
                y_coord = yyInterpol[i]

                if (0 <= x_coord < sPSF[1]-1) and (0 <= y_coord < sPSF[0]-1):
                    flux_val = float(interp_func(y_coord, x_coord)[0, 0])
                    flux_percent.append(flux_val)

            flux_percent = np.array(flux_percent)

        else:
            raise ValueError(f"FROM_PSF sampling type {self.sampling_type} not implemented")

        # Normalize flux
        flux_percent = flux_percent / np.sum(flux_percent)

        # Convert to tip/tilt coefficients
        coeff_tiltx = self._angle_to_tip(xx_arcsec)
        coeff_tilty = self._angle_to_tip(yy_arcsec)
        coeff_focus = np.zeros(len(xx_arcsec))

        return {
            'xx_arcsec': xx_arcsec,
            'yy_arcsec': yy_arcsec,
            'coeff_tiltx': coeff_tiltx,
            'coeff_tilty': coeff_tilty,
            'coeff_focus': coeff_focus,
            'coeff_flux': flux_percent
        }

    def _compute_3d(self) -> dict:
        """Compute 3D extended source"""
        if not self.layer_height or not self.intensity_profile:
            raise ValueError("3D source requires layer_height and intensity_profile")

        n_layers = len(self.layer_height)
        if len(self.intensity_profile) != n_layers:
            raise ValueError("layer_height and intensity_profile must have same length")

        # Initialize arrays
        xx_arcsec_all = []
        yy_arcsec_all = []
        coeff_tiltx_all = []
        coeff_tilty_all = []
        coeff_focus_all = []
        coeff_flux_all = []

        # Process each layer
        for i in range(n_layers):
            # Compute focus coefficient for this layer
            focus_coeff = self._compute_focus_coefficient(self.layer_height[i])

            # Compute 2D pattern for this layer
            layer_result = self._compute_2d()

            # Add tip/tilt offset if specified
            tt_offset = self.tt_profile[i] if self.tt_profile is not None else [0.0, 0.0]

            # Accumulate results
            xx_arcsec_all.extend(layer_result['xx_arcsec'] + tt_offset[0])
            yy_arcsec_all.extend(layer_result['yy_arcsec'] + tt_offset[1])
            coeff_tiltx_all.extend(layer_result['coeff_tiltx'] + self._angle_to_tip(tt_offset[0]))
            coeff_tilty_all.extend(layer_result['coeff_tilty'] + self._angle_to_tip(tt_offset[1]))
            coeff_focus_all.extend(np.full(len(layer_result['coeff_focus']), focus_coeff))
            coeff_flux_all.extend(layer_result['coeff_flux'] * self.intensity_profile[i])

        return {
            'xx_arcsec': np.array(xx_arcsec_all),
            'yy_arcsec': np.array(yy_arcsec_all),
            'coeff_tiltx': np.array(coeff_tiltx_all),
            'coeff_tilty': np.array(coeff_tilty_all),
            'coeff_focus': np.array(coeff_focus_all),
            'coeff_flux': np.array(coeff_flux_all)
        }

    def _angle_to_tip(self, angle_arcsec: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert angle in arcsec to tip/tilt coefficient in rad RMS @ wavelength"""
        # From IDL: angle2tip function
        sec2rad = 4.848e-6
        angle_rad = angle_arcsec * sec2rad
        return angle_rad * self.d_tel / (self.wavelengthInNm * 1e-9) / (2.0 * np.pi)

    def _compute_focus_coefficient(self, layer_height: float) -> float:
        """Compute focus coefficient for a layer at given height"""
        if self.focus_height is None:
            return 0.0

        delta_height = layer_height - self.focus_height
        focal_ratio = layer_height / self.d_tel

        # From IDL formula
        focus_coeff = delta_height / (2.0 * np.sqrt(3.0) * 8.0 * focal_ratio**2)
        focus_coeff *= (2.0 * np.pi) / (self.wavelengthInNm * 1e-9)

        return focus_coeff

    def _apply_flux_threshold(self):
        """Apply flux threshold to remove low-intensity points"""
        mean_flux = np.mean(self.coeff_flux)
        threshold = mean_flux * self.flux_threshold
        valid_mask = self.coeff_flux > threshold

        self.xx_arcsec = self.xx_arcsec[valid_mask]
        self.yy_arcsec = self.yy_arcsec[valid_mask]
        self.coeff_tiltx = self.coeff_tiltx[valid_mask]
        self.coeff_tilty = self.coeff_tilty[valid_mask]
        self.coeff_focus = self.coeff_focus[valid_mask]
        self.coeff_flux = self.coeff_flux[valid_mask]

        self.npoints = len(self.coeff_flux)

    def trigger(self):
        """Update PSF if new data is available and recompute if needed"""
        if self.source_type == 'FROM_PSF':
            psf = self.local_inputs.get('psf')
            if np.sum(self.xp.abs(psf.value)) > 0:
                self.psf = psf
                self.compute()  # Recompute all coefficients with new PSF

    def plot_source(self):
        """Plot the extended source distribution"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))

            # Spatial distribution
            plt.subplot(131)
            scatter = plt.scatter(self.xx_arcsec, self.yy_arcsec, 
                                c=self.coeff_flux, s=50, cmap='viridis')
            plt.colorbar(scatter, label='Flux')
            plt.xlabel('X [arcsec]')
            plt.ylabel('Y [arcsec]')
            plt.title(f'{self.source_type} - {self.sampling_type}\n{self.npoints} points')
            plt.axis('equal')

            # Tip coefficients
            plt.subplot(132)
            plt.scatter(self.coeff_tiltx, self.coeff_tilty, 
                    c=self.coeff_flux, s=50, cmap='viridis')
            plt.xlabel('Tip coefficient [rad]')
            plt.ylabel('Tilt coefficient [rad]')
            plt.title('Tip/Tilt coefficients')

            # Focus distribution
            plt.subplot(133)
            plt.hist(self.coeff_focus, bins=20, alpha=0.7)
            plt.xlabel('Focus coefficient [rad]')
            plt.ylabel('Count')
            plt.title('Focus distribution')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")