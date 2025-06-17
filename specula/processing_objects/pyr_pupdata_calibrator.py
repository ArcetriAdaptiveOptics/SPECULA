import os
import numpy as np
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.pupdata import PupData

class PyrPupdataCalibrator(BaseProcessingObj):
    def __init__(self,
                 data_dir: str,
                 thr1: float = 0.1,
                 thr2: float = 0.25,
                 output_tag: str = None,
                 auto_detect_obstruction: bool = True,
                 min_obstruction_ratio: float = 0.05,
                 display_debug: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.thr1 = thr1
        self.thr2 = thr2
        self.auto_detect_obstruction = auto_detect_obstruction
        self.min_obstruction_ratio = min_obstruction_ratio
        self.display_debug = display_debug
        self._data_dir = data_dir
        self._filename = output_tag or "pupdata"
        self.central_obstruction_ratio = 0.0

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
            ind_pup=ind_pup[pup_order, :],
            radius=radii[pup_order],
            cx=centers[pup_order, 0],
            cy=centers[pup_order, 1],
            framesize=image.shape
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
        offsets = [[cx-dim, cy-dim], [cx, cy-dim], [cx-dim, cy], [cx, cy]]

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
            if len(profile) > 5:
                center_intensity = self.xp.mean(profile[:3])  # Inner 3 bins
                edge_intensity = self.xp.mean(profile[-3:])   # Outer 3 bins

                if edge_intensity > center_intensity * 1.5:  # 50% intensity drop
                    # Find where intensity starts rising
                    grad = self.xp.gradient(profile)
                    max_grad_idx = self.xp.argmax(grad[:len(grad)//2])  # First half only
                    obstruction_ratio = (max_grad_idx / len(profile)) * 0.8  # Conservative

                    if obstruction_ratio >= self.min_obstruction_ratio:
                        obstruction_ratios.append(obstruction_ratio)

        return self.xp.median(obstruction_ratios) if obstruction_ratios else 0.0

    def _radial_profile(self, image, center, max_radius, n_bins=20):
        """Extract radial intensity profile"""
        h, w = image.shape
        y, x = self.xp.mgrid[0:h, 0:w]
        r = self.xp.sqrt((x - center[0])**2 + (y - center[1])**2)

        profile = []
        for i in range(n_bins):
            r_inner = (i / n_bins) * max_radius
            r_outer = ((i + 1) / n_bins) * max_radius
            mask = (r >= r_inner) & (r < r_outer)
            if self.xp.any(mask):
                profile.append(self.xp.mean(image[mask]))
            else:
                profile.append(0)

        return self.xp.array(profile)

    def _generate_indices(self, centers, radii, image_shape):
        """Generate pupil pixel indices with optional obstruction"""
        h, w = image_shape
        y_coords, x_coords = self.xp.mgrid[0:h, 0:w]

        # Estimate max pixels per pupil
        max_pixels = int(self.xp.pi * self.xp.max(radii)**2 * (1 - self.central_obstruction_ratio**2)) + 100
        ind_pup = self.xp.zeros((4, max_pixels), dtype=int)

        for i in range(4):
            if radii[i] <= 0:
                continue

            # Distance from center
            r = self.xp.sqrt((x_coords - centers[i, 0])**2 + (y_coords - centers[i, 1])**2)

            # Create mask (annulus if obstruction detected)
            if self.central_obstruction_ratio > 0:
                mask = (r <= radii[i]) & (r >= radii[i] * self.central_obstruction_ratio)
            else:
                mask = r <= radii[i]

            # Get flat indices
            flat_indices = self.xp.where(mask.flatten())[0]
            n_pixels = min(len(flat_indices), max_pixels)

            ind_pup[i, :n_pixels] = flat_indices[:n_pixels]
            if n_pixels < max_pixels:
                ind_pup[i, n_pixels:] = flat_indices[0] if n_pixels > 0 else 0

        return ind_pup

    def _debug_plot(self, image, centers, radii):
        """Simple debug plot"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle

            plt.figure(figsize=(10, 5))

            # Image with circles
            plt.subplot(1, 2, 1)
            plt.imshow(image, origin='lower', cmap='gray')

            colors = ['red', 'green', 'blue', 'orange']
            for i, (center, radius) in enumerate(zip(centers, radii)):
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
                plt.plot(profile, 'b-', linewidth=2)
                if self.central_obstruction_ratio > 0:
                    obs_idx = int(len(profile) * self.central_obstruction_ratio)
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

        self.pupdata.save(file_path)

        if self.verbose:
            print(f'Saved pupil data: {file_path}')
            print(f'Obstruction ratio: {self.central_obstruction_ratio:.3f}')