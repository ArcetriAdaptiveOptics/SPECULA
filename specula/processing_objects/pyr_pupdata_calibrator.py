import os

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.data_objects.intensity import Intensity
from specula.data_objects.pixels import Pixels
from specula.connections import InputValue
from specula.data_objects.pupdata import PupData
from specula.lib import utils
from specula import cpuArray


class PyrPupdataCalibrator(BaseProcessingObj):
    """
    Pyramid PupData Calibrator processing object. Calibrator for pyramid pupils.

    This processing object analyzes a calibration image containing four
    pyramid pupils to estimate their geometric properties (centers and radii),
    detect possible central obstructions, and generate pixel index maps for
    each pupil. The resulting data is stored in a :class:`PupData` object.

    Optional features include automatic central obstruction detection and debug plotting.

    The calibration can operate on either intensity or pixel inputs and supports
    optional temporal integration, obstruction detection, and debug visualization.

    Inputs
    ------
    in_i : Intensity, optional
        Input intensity data.
    in_pixels : Pixels, optional
        Input pixel data.

    Outputs
    -------
    out_pupdata : PupData
        Extracted pupil data, including centers, radii, and pixel indices.

    Notes
    -----
    - At least one input (``in_i`` or ``in_pixels``) must be provided.
    - Pupil ordering is rearranged to match PASSATA conventions.
    """

    def __init__(self,
                 data_dir: str,      # Set by main Simul object
                 dt: float = None,
                 thr1: float = 0.1,
                 thr2: float = 0.25,
                 obs_thr: float = 0.8,
                 slopes_from_intensity: bool=False,
                 output_tag: str = None,
                 auto_detect_obstruction: bool = True,
                 min_obstruction_ratio: float = 0.05,
                 display_debug: bool = False,
                 overwrite: bool = False,
                 save_on_exit: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters
        ----------
        data_dir : str
            Directory where calibration outputs are saved.
        dt : float, optional
            Integration time in seconds. If provided, frames are accumulated and
            processed only at multiples of ``dt``.
        thr1 : float, optional
            First threshold for pupil segmentation. Default is 0.1.
        thr2 : float, optional
            Second threshold for refined segmentation. Default is 0.25.
        obs_thr : float, optional
            Scaling factor for obstruction detection. Default is 0.8.
        slopes_from_intensity : bool, optional
            If True, generate pupil indices directly from intensity masks.
            Otherwise, use geometric translation. Default is False.
        output_tag : str, optional
            Filename used when saving calibration results.
        auto_detect_obstruction : bool, optional
            Enable automatic detection of central obstruction. Default is True.
        min_obstruction_ratio : float, optional
            Minimum allowed obstruction ratio. Default is 0.05.
        display_debug : bool, optional
            If True, display debug plots during calibration. Default is False.
        overwrite : bool, optional
            If True, overwrite existing files when saving. Default is False.
        save_on_exit : bool, optional
            If True, automatically save calibration data on finalize. Default is True.
        target_device_idx : int, optional
            Target device index for computation.
        precision : int, optional
            Numerical precision for internal data.

        Raises
        ------
        ValueError
            If ``dt`` is provided and is not positive.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)


        if dt is not None:   
            if dt <= 0:
                raise ValueError(f'dt (integration time) is {dt} and must be greater than zero')
            self.dt = self.seconds_to_t(dt)
        else:
            self.dt = None

        self.thr1 = thr1
        self.thr2 = thr2
        self.obs_thr = obs_thr
        self.slopes_from_intensity = slopes_from_intensity
        self.auto_detect_obstruction = auto_detect_obstruction
        self.min_obstruction_ratio = min_obstruction_ratio
        self.display_debug = display_debug
        self.data_dir = data_dir
        self.filename = output_tag
        self.central_obstruction_ratio = 0.0
        self.overwrite = overwrite
        self.save_on_exit = save_on_exit
        self.integrated_pixels = None

        # Outputs
        self.pupdata = PupData(
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        self.outputs['out_pupdata'] = self.pupdata

        # Inputs
        self.inputs['in_i'] = InputValue(type=Intensity, optional=True)  
        self.inputs['in_pixels'] = InputValue(type=Pixels, optional=True)

    @classmethod
    def input_names(cls):
        return {'in_i': InputDesc(Intensity, 'Input intensity image (optional)'),
                'in_pixels': InputDesc(Pixels, 'Input pixel image (optional)')}

    @classmethod
    def output_names(cls):
        return {'out_pupdata': OutputDesc(PupData, 'Calibrated pupil data with subaperture geometry')}

    def setup(self):
        super().setup()
        if self.local_inputs['in_i'] is None and self.local_inputs['in_pixels'] is None:
            raise ValueError("At least one input must be provided for calibration. ")

    def trigger_code(self):
        """Main calibration function"""

        if self.local_inputs['in_i'] is not None:
            value = self.local_inputs['in_i'].i
        elif self.local_inputs['in_pixels'] is not None:
            value = self.local_inputs['in_pixels'].pixels

        # Integrate pixels or intensity over time
        if self.integrated_pixels is None:
            self.integrated_pixels = value * 0

        self.integrated_pixels += value

        # if dt is set, only trigger on multiples of dt, otherwise trigger on every frame
        if self.dt is not None:
            if self.current_time % self.dt != 0:
                return

        image = self.integrated_pixels

        # Analyze pupils
        centers, radii = self._analyze_pupils(image)

        # Auto-detect obstruction
        if self.auto_detect_obstruction:
            self.central_obstruction_ratio = self._detect_obstruction(image, centers, radii)

        # Debug plot
        if self.display_debug:
            self._debug_plot(image, centers, radii)

        # Generate indices
        ind_pup = self._generate_indices(centers, radii, image.shape)

        # Create PupData (reorder to match IDL)
        pup_order = [1, 0, 2, 3]
        self.pupdata.ind_pup = ind_pup[:, pup_order]
        self.pupdata.radius = radii[pup_order]
        self.pupdata.cx = centers[pup_order, 0]
        self.pupdata.cy = centers[pup_order, 1]
        self.pupdata.framesize = image.shape
        self.pupdata.slopes_from_intensity = self.slopes_from_intensity
        self.pupdata.generation_time = self.current_time

        # Reset integrated intensity
        self.integrated_pixels *= 0.0

    def _analyze_pupils(self, image):
        """
        Detect pupil centers and radii in four quadrants.

        Parameters
        ----------
        image : array-like
            Input image.

        Returns
        -------
        centers : ndarray of shape (4, 2)
            Estimated (x, y) coordinates of pupil centers.
        radii : ndarray of shape (4,)
            Estimated pupil radii.
        """
        h, w = image.shape
        cy, cx = h // 2, w // 2
        dim = min(cx, cy)

        # Extract 4 quadrants
        quadrants = [
            image[cy-dim:cy, cx-dim:cx],     # Top-left
            image[cy-dim:cy, cx:cx+dim],     # Top-right
            image[cy:cy+dim, cx-dim:cx],     # Bottom-left
            image[cy:cy+dim, cx:cx+dim]      # Bottom-right
        ]

        # Quadrant offsets
        offsets = self.xp.array([[cx-dim, cy-dim], [cx, cy-dim], [cx-dim, cy], [cx, cy]])

        centers = self.xp.zeros((4, 2))
        radii = self.xp.zeros(4)

        for i, (quad, offset) in enumerate(zip(quadrants, offsets)):
            center, radius = self._analyze_single_pupil(quad)
            centers[i] = center + offset
            radii[i] = radius

        return centers, radii

    def _analyze_single_pupil(self, image):
        """
        Analyze a single pupil region.

        Uses a two-level thresholding approach to extract a binary mask,
        then computes centroid and radius.

        Parameters
        ----------
        image : array-like
            Input quadrant image.

        Returns
        -------
        center : ndarray of shape (2,)
            Estimated (x, y) center. These coordinates will be 0 if no valid region is found.
        radius : float
            Estimated radius. Returns 0 if no valid region is found.
        """
        # Two-level thresholding
        min_val, max_val = float(self.xp.min(image)), float(self.xp.max(image))
        s1 = min_val + (max_val - min_val) * self.thr1

        thresh_img = image.copy()
        thresh_img[thresh_img < s1] = 0

        s2 = float(self.xp.mean(thresh_img[thresh_img > 0])) * self.thr2
        mask = thresh_img >= s2

        # Calculate centroid and radius
        if self.xp.any(mask):
            y_coords, x_coords = self.xp.mgrid[0:image.shape[0], 0:image.shape[1]]
            x_center = self.xp.sum(x_coords * mask) / self.xp.sum(mask)
            y_center = self.xp.sum(y_coords * mask) / self.xp.sum(mask)
            radius = self.xp.sqrt(self.xp.sum(mask) / self.xp.pi)
            return self.xp.array([x_center, y_center]), radius
        else:
            return self.xp.array([0.0, 0.0]), 0.0

    def _detect_obstruction(self, image, centers, radii):
        """
        Estimate central obstruction ratio.

        Parameters
        ----------
        image : array-like
            Input image.
        centers : ndarray
            Pupil centers.
        radii : ndarray
            Pupil radii.

        Returns
        -------
        float
            Estimated obstruction ratio (0 if none detected).

        Notes
        -----
        Detection is based on radial intensity profiles and gradient analysis.
        """
        obstruction_ratios = []

        for i in range(4):
            if radii[i] <= 0:
                continue

            # Extract radial profile
            profile = self._radial_profile(image, centers[i], radii[i])

            # Look for central dip
            if profile.shape[0] > 5:
                center_intensity = self.xp.mean(profile[:3])  # Inner 3 bins
                edge_intensity = self.xp.mean(profile[-3:])   # Outer 3 bins

                if edge_intensity > center_intensity * 1.5:  # 50% intensity drop
                    # Find where intensity starts rising
                    grad = self.xp.gradient(profile)
                    max_grad_idx = self.xp.argmax(grad[:grad.shape[0]//2])  # First half only
                    obstruction_ratio = (float(max_grad_idx) / profile.shape[0]) * self.obs_thr

                    if obstruction_ratio >= self.min_obstruction_ratio:
                        obstruction_ratios.append(float(obstruction_ratio))
                    else:
                        obstruction_ratios.append(self.min_obstruction_ratio)

        if obstruction_ratios:
            return float(self.xp.median(self.xp.array(obstruction_ratios)))
        else:
            return 0.0

    def _radial_profile(self, image, center, max_radius, n_bins=20):
        """
        Compute radial intensity profile.

        Parameters
        ----------
        image : array-like
            Input image.
        center : array-like
            Pupil center (x, y).
        max_radius : float
            Maximum radius to consider.
        n_bins : int, optional
            Number of radial bins. Default is 20.

        Returns
        -------
        ndarray
            Radial intensity profile.
        """
        h, w = image.shape
        y, x = self.xp.mgrid[0:h, 0:w]
        r = self.xp.sqrt((x - center[0])**2 + (y - center[1])**2)

        profile = self.xp.zeros(n_bins)
        for i in range(n_bins):
            r_inner = (i / n_bins) * max_radius
            r_outer = ((i + 1) / n_bins) * max_radius
            mask = (r >= r_inner) & (r < r_outer)
            if self.xp.any(mask):
                profile[i] = self.xp.mean(image[mask])

        return self.xp.array(profile)

    def _generate_indices(self, centers, radii, image_shape):
        """
        Generate pixel indices for each pupil.

        Parameters
        ----------
        centers : ndarray
            Pupil centers.
        radii : ndarray
            Pupil radii.
        image_shape : tuple
            Shape of the input image.

        Returns
        -------
        ndarray
            Array of pixel indices for each pupil.

        Raises
        ------
        ValueError
            If no valid pupils are detected or indices cannot be generated.

        Depending on the "self.slopes_from_intensity" parameter, indices are either
        calculated for each pupil independently (if the parameter is True), or derived
        from a single pupil mask that is replicated four times and translated to match
        each pupil (if the parameter is False).
        """
        h, w = image_shape
        y_coords, x_coords = self.xp.mgrid[0:h, 0:w]

        if self.slopes_from_intensity:
            # INTENSITY MODE: Adapt to real indices of each pupil
            # Compute maximum number of pixels needed
            max_pixels = 0
            temp_indices = []

            for i in range(4):
                if radii[i] <= 0:
                    raise ValueError("Invalid radius detected on index {i}. "
                                     "Check input image and parameters.")

                # Distance from center
                r = self.xp.sqrt((x_coords - centers[i, 0])**2 + (y_coords - centers[i, 1])**2)

                # Create mask (annulus if obstruction detected)
                if self.central_obstruction_ratio > 0:
                    mask = (r <= radii[i]) & (r >= radii[i] * self.central_obstruction_ratio)
                else:
                    mask = r <= radii[i]

                # Get flat indices
                flat_indices = self.xp.where(mask.flatten())[0]
                temp_indices.append(flat_indices)
                max_pixels = max(max_pixels, flat_indices.shape[0])

            # Create a 2D array with padding to -1
            ind_pup = self.xp.full((max_pixels, 4), -1, dtype=int)

            for i, indices in enumerate(temp_indices):
                if indices.shape[0] > 0:
                    ind_pup[:indices.shape[0], i] = indices

        else:
            # SLOPES MODE: Identical areas obtained by simple translation
            # Use first valid pupil as reference geometry
            valid_pupils = [i for i in range(4) if radii[i] > 0]
            if len(valid_pupils) != 4:
                raise ValueError("All four pupils must be valid (radius > 0) for geometric mode. "
                                f"Found valid pupils: {valid_pupils}")

            # Create reference mask using first valid pupil and maximum radius
            reference_pupil_idx = valid_pupils[0]
            ref_center = centers[reference_pupil_idx]
            ref_radius = radii.max()

            r_ref = self.xp.sqrt((x_coords - ref_center[0])**2 + (y_coords - ref_center[1])**2)

            if self.central_obstruction_ratio > 0:
                reference_mask = (r_ref <= ref_radius) & (r_ref >= ref_radius * self.central_obstruction_ratio)
            else:
                reference_mask = r_ref <= ref_radius

            # Find relative offsets from reference center
            reference_indices = self.xp.where(reference_mask.flatten())[0]
            ref_y, ref_x = self.xp.unravel_index(reference_indices, (h, w))

            # Calculate relative offsets from reference pupil center
            offset_x = ref_x - ref_center[0]
            offset_y = ref_y - ref_center[1]

            # Number of pixels per pupil (same for all)
            n_pixels = reference_indices.shape[0]
            ind_pup = self.xp.full((n_pixels, 4), -1, dtype=int)

            # Apply translation for each pupil
            for i in range(4):

                # Calculate INTEGER translation vector
                translation_x = self.xp.round(centers[i, 0] - ref_center[0]).astype(int)
                translation_y = self.xp.round(centers[i, 1] - ref_center[1]).astype(int)

                # Apply integer translation to the reference geometry
                new_x = offset_x + ref_center[0] + translation_x  # Same as: offset_x + centers[i,0] rounded
                new_y = offset_y + ref_center[1] + translation_y  # Same as: offset_y + centers[i,1] rounded

                # Check that pixels are inside the image
                valid_mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)

                # Raise error if no pixels are inside the image
                if not self.xp.any(valid_mask):
                    raise ValueError(f"Pupil {i} after translation is completely outside the image bounds. "
                                    f"Translation: ({translation_x}, {translation_y}), "
                                    f"Image size: {w}x{h}")

                # Convert to linear indices
                valid_x = new_x[valid_mask].astype(int)
                valid_y = new_y[valid_mask].astype(int)
                valid_linear_indices = valid_y * w + valid_x

                # Fill array with valid indices
                n_valid = valid_linear_indices.shape[0]
                ind_pup[:n_valid, i] = valid_linear_indices

                # Warn if any pixel is lost
                lost_pixels = n_pixels - n_valid
                if lost_pixels > 0:
                    print(f"Warning: Pupil {i} lost {lost_pixels}/{n_pixels} pixels "
                        f"({100*lost_pixels/n_pixels:.1f}%) due to image boundaries")

                # If fewer valid pixels, pad with -1 (already done by xp.full)

        # Look for any rows with all -1 and remove them
        valid_rows = self.xp.any(ind_pup != -1, axis=1)
        # If no valid rows raise an error
        if not self.xp.any(valid_rows):
            raise ValueError("No valid pupil indices found. Check input image and parameters.")
        # Filter out invalid rows
        ind_pup = ind_pup[valid_rows]

        return ind_pup

    def _debug_plot(self, image, centers, radii):
        """
        Display debug visualization of detected pupils.

        Parameters
        ----------
        image : array-like
            Input image.
        centers : ndarray
            Pupil centers.
        radii : ndarray
            Pupil radii.

        Notes
        -----
        Requires ``matplotlib``. If unavailable, a warning is printed.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            plt.figure(figsize=(10, 5))

            # Convert to CPU arrays for matplotlib
            image_cpu = cpuArray(image)
            centers_cpu = cpuArray(centers)
            radii_cpu = cpuArray(radii)

            # Image with circles
            plt.subplot(1, 2, 1)
            plt.imshow(image_cpu, origin='lower', cmap='gray')

            colors = ['red', 'green', 'blue', 'orange']
            for i, (center, radius) in enumerate(zip(centers_cpu, radii_cpu)):
                if radius > 0:
                    circle = Circle(center, radius, fill=False, color=colors[i], linewidth=2)
                    plt.gca().add_patch(circle)

                    if self.central_obstruction_ratio > 0:
                        obs_circle = Circle(center, radius * self.central_obstruction_ratio, 
                                          fill=False, color=colors[i], linestyle='--')
                        plt.gca().add_patch(obs_circle)

            plt.title(f'Detected Pupils (obstruction: {self.central_obstruction_ratio:.3f})')

            # Radial profile example
            plt.subplot(1, 2, 2)
            if radii[0] > 0:
                profile = self._radial_profile(image, centers[0], radii[0])
                profile_cpu = cpuArray(profile)
                plt.plot(profile_cpu, 'b-', linewidth=2)
                if self.central_obstruction_ratio > 0:
                    obs_idx = int(profile_cpu.shape[0] * self.central_obstruction_ratio)
                    plt.axvline(obs_idx, color='red', linestyle='--', label='Obstruction')
                plt.title('Radial Profile (Pupil 0)')
                plt.xlabel('Radial bin')
                plt.ylabel('Intensity')
                plt.legend()

            plt.tight_layout()
            plt.show(block=True)
            plt.pause(0.1)

        except ImportError:
            print("Matplotlib not available for debug plotting")

    def _save(self, filename):
        """
        Save pupil calibration data to disk.

        Parameters
        ----------
        filename : str
            Output filename (FITS format). If no extension is provided,
            ``.fits`` is appended.

        Raises
        ------
        ValueError
            If no pupil data is available to save.
        """
        if filename is None:
            filename = utils.make_tn()

        if self.pupdata is None:
            raise ValueError("No pupil data to save")

        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self.data_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.pupdata.save(file_path, overwrite=self.overwrite)

        if self.verbose:
            print(f'Saved pupil data: {file_path}')
            print(f'Obstruction ratio: {self.central_obstruction_ratio:.3f}')

    def finalize(self):
        if self.save_on_exit:
            self._save(self.filename)

