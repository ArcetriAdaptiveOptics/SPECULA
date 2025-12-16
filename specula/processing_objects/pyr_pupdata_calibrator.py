import os
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.pupdata import PupData
from specula import cpuArray


class PyrPupdataCalibrator(BaseProcessingObj):
    def __init__(self,
                 data_dir: str,
                 thr1: float = 0.1,
                 thr2: float = 0.25,
                 obs_thr: float = 0.8,
                 slopes_from_intensity: bool=False,
                 output_tag: str = None,
                 auto_detect_obstruction: bool = True,
                 min_obstruction_ratio: float = 0.05,
                 display_debug: bool = False,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.thr1 = thr1
        self.thr2 = thr2
        self.obs_thr = obs_thr
        self.slopes_from_intensity = slopes_from_intensity
        self.auto_detect_obstruction = auto_detect_obstruction
        self.min_obstruction_ratio = min_obstruction_ratio
        self.display_debug = display_debug
        self._data_dir = data_dir
        self._filename = output_tag or "pupdata"
        self.central_obstruction_ratio = 0.0
        self._overwrite = overwrite

        self.inputs['in_i'] = InputValue(type=Intensity)
        self.pupdata = None

    def trigger_code(self):
        """Main calibration function"""
        image = self.local_inputs['in_i'].i

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
        self.pupdata = PupData(
            ind_pup=ind_pup[:, pup_order],
            radius=radii[pup_order],
            cx=centers[pup_order, 0],
            cy=centers[pup_order, 1],
            framesize=image.shape,
            target_device_idx=self.target_device_idx
        )

    def _analyze_pupils(self, image):
        """Find 4 pupil centers and radii"""
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
        """Analyze single pupil quadrant"""
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
        """Simple obstruction detection"""
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

        if obstruction_ratios:
            return float(self.xp.median(self.xp.array(obstruction_ratios)))
        else:
            return 0.0

    def _radial_profile(self, image, center, max_radius, n_bins=20):
        """Extract radial intensity profile"""
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
        """Generate pupil pixel indices with optional obstruction"""
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
        """Simple debug plot"""
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

    def finalize(self):
        """Save pupil data"""
        if self.pupdata is None:
            raise ValueError("No pupil data to save")

        filename = self._filename
        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self._data_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.pupdata.save(file_path, overwrite=self._overwrite)

        if self.verbose:
            print(f'Saved pupil data: {file_path}')
            print(f'Obstruction ratio: {self.central_obstruction_ratio:.3f}')
