import os
import numpy as np
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.pupdata import PupData
from specula.data_objects.simul_params import SimulParams


class PyrPupdataCalibrator(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 thr1: float = 0.1,           # Threshold per background removal
                 thr2: float = 0.25,          # Threshold per pupil refinement
                 output_tag: str = None,
                 tag_template: str = None,
                 do_not_ave_pup_cen: bool = False,    # Avoid averaging pupil centers
                 do_pup_inter_or_union: str = 'Union', # 'Union', 'Inter', or 'None'
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.thr1 = thr1
        self.thr2 = thr2
        self.do_not_ave_pup_cen = do_not_ave_pup_cen
        self.do_pup_inter_or_union = do_pup_inter_or_union
        self._data_dir = simul_params.root_dir

        if tag_template is None and (output_tag is None or output_tag == 'auto'):
            raise ValueError('At least one of tag_template and output_tag must be set')

        if output_tag is None or output_tag == 'auto':
            self._filename = tag_template
        else:
            self._filename = output_tag

        self.inputs['in_i'] = InputValue(type=Intensity)
        self.pupdata = None

    def trigger_code(self):
        """Main trigger code - equivalent to pupil_acquire IDL function"""
        image = self.local_inputs['in_i'].i

        # Ensure even dimensions (equivalent to IDL dimension adjustment)
        image = self._ensure_even_dimensions(image)

        # Analyze the four pupils
        centers, radii = self._analyze_pupils(image)

        if self.verbose:
            print(f'Found pupil centers: {centers}')
            print(f'Found pupil radii: {radii}')

        # Refine pupil centers and radii
        new_radii, new_centers = self._refine_pup_centers(radii, centers)

        if self.verbose:
            print(f'Refined pupil centers: {new_centers}')
            print(f'Refined pupil radii: {new_radii}')

        # Generate pupil indices
        ind_pup = self._generate_pupil_indices(new_radii, new_centers, image.shape)

        # Create PupData object with reordered pupils (matching IDL pup_order = [3,2,0,1])
        pup_order = [3, 2, 0, 1]
        self.pupdata = PupData(target_device_idx=self.target_device_idx, precision=self.precision)
        self.pupdata.ind_pup = ind_pup[pup_order, :]
        self.pupdata.radius = new_radii[pup_order]
        self.pupdata.cx = new_centers[pup_order, 0] 
        self.pupdata.cy = new_centers[pup_order, 1]
        self.pupdata.framesize = self.xp.array(image.shape, dtype=int)

    def _ensure_even_dimensions(self, image):
        """Ensure image has even dimensions (from IDL code)"""
        h, w = image.shape
        new_h = h if h % 2 == 0 else h + 1
        new_w = w if w % 2 == 0 else w + 1

        if new_h != h or new_w != w:
            if self.verbose:
                print(f'Adjusting dimensions from {image.shape} to ({new_h}, {new_w})')
            new_image = self.xp.zeros((new_h, new_w), dtype=image.dtype)
            new_image[:h, :w] = image
            return new_image
        return image

    def _analyze_pupils(self, image):
        """Equivalent to pyr_distanza_centri + pyr_analizza IDL functions"""
        # Set border pixels to zero
        image = image.copy()
        image[0, :] = 0
        image[-1, :] = 0
        image[:, 0] = 0
        image[:, -1] = 0

        h, w = image.shape
        cx, cy = h // 2, w // 2

        # Split image into 4 quadrants (equivalent to IDL SPLIT logic)
        dim = min(cx, cy)
       
        # Create 4 pupil subimages
        reduce = 0  # Could be cx//20 for ccd39 fix
        cx += reduce
        dim -= reduce

        pupils = self.xp.zeros((4, dim, dim))
        pupils[0] = image[cx-dim:cx, cy:cy+dim]       # Top-left
        pupils[1] = image[cx:cx+dim, cy:cy+dim]      # Top-right  
        pupils[2] = image[cx-dim:cx, cy-dim:cy]      # Bottom-left
        pupils[3] = image[cx:cx+dim, cy-dim:cy]      # Bottom-right

        centers = self.xp.zeros((4, 2))
        radii = self.xp.zeros(4)

        for i in range(4):
            if self.verbose:
                print(f'Analyzing pupil {i}')

            pupil_img = pupils[i].copy()
            center, radius = self._analyze_single_pupil(pupil_img)

            # Correct coordinates for quadrant position
            if i == 0:    # Top-left
                center += [cx-dim, cy]
            elif i == 1:  # Top-right
                center += [cx, cy]
            elif i == 2:  # Bottom-left  
                center += [cx-dim, cy-dim]
            elif i == 3:  # Bottom-right
                center += [cx, cy-dim]

            centers[i] = center + 0.5  # IDL AGGIUSTINO2
            radii[i] = radius

            if self.verbose:
                print(f'Pupil {i}: Diameter = {2*radius:.1f}, Center = {center}')

        return centers, radii

    def _analyze_single_pupil(self, pupil_img):
        """Equivalent to pyr_analizza IDL function for single pupil"""
        # First threshold (background removal)
        min_val = float(self.xp.min(pupil_img))
        max_val = float(self.xp.max(pupil_img))
        s1 = min_val + (max_val - min_val) * self.thr1

        pupil_thresh = pupil_img.copy()
        pupil_thresh[pupil_thresh < s1] = 0

        # Second threshold
        mean_val = float(self.xp.mean(pupil_thresh))
        s2 = mean_val * self.thr2
        pupil_thresh[pupil_thresh < s2] = 0

        if self.verbose:
            print(f'  Thresholds: s1={s1:.1f}, s2={s2:.1f}')

        # Iterative refinement (equivalent to IDL repeat-until loop)
        max_iterations = 10
        for iteration in range(max_iterations):
            # Calculate centroid
            center = self._calculate_centroid(pupil_thresh)

            # Calculate radius 
            radius = self._calculate_radius(pupil_thresh, s2)

            # Apply threshold inside calculated radius
            pixels_changed = self._apply_threshold_in_radius(pupil_thresh, s2, radius, center)

            if self.verbose:
                print(f'  Iteration {iteration}: center={center}, radius={radius:.1f}')

            # Stop when no more pixels change
            if pixels_changed == 0:
                break

        return center, radius

    def _calculate_centroid(self, image):
        """Equivalent to calcola_baricentro IDL function"""
        h, w = image.shape
        y_indices, x_indices = self.xp.mgrid[0:h, 0:w]

        mask = image > 0
        count = self.xp.sum(mask)

        if count > 0:
            y_center = self.xp.sum(y_indices * mask) / count
            x_center = self.xp.sum(x_indices * mask) / count
            return self.xp.array([x_center, y_center])
        else:
            return self.xp.array([0.0, 0.0])

    def _calculate_radius(self, image, threshold):
        """Equivalent to calcola_raggio IDL function"""
        pixel_count = int(self.xp.sum(image >= threshold))
        area = pixel_count / self.xp.pi
        return float(self.xp.sqrt(area))

    def _apply_threshold_in_radius(self, image, threshold, radius, center, threshold_margin=0.1):
        """Equivalent to sottosoglia IDL function"""
        h, w = image.shape
        y_indices, x_indices = self.xp.mgrid[0:h, 0:w]

        # Distance from center
        dx = x_indices - center[0]
        dy = y_indices - center[1] 
        distance_sq = dx**2 + dy**2

        # Pixels inside radius (with margin)
        inside_radius = distance_sq < (radius - threshold_margin)**2

        # Count changed pixels
        old_values = image[inside_radius].copy()
        image[inside_radius] = threshold
        changed_pixels = self.xp.sum(old_values != threshold)

        return changed_pixels

    def _refine_pup_centers(self, radii, centers):
        """Equivalent to refine_pup_centers IDL function"""
        # Use minimum radius for all pupils
        new_radius = float(self.xp.min(radii))
        new_radii = self.xp.full(4, new_radius)
        new_centers = centers.copy()

        if not self.do_not_ave_pup_cen:
            # Original averaging logic
            max_x = float(self.xp.max(centers[:, 0]))
            min_x = float(self.xp.min(centers[:, 0]))
            max_y = float(self.xp.max(centers[:, 1]))
            min_y = float(self.xp.min(centers[:, 1]))
            
            distance = float(self.xp.round(self.xp.mean([max_x - min_x, max_y - min_y])))

            # Find pupils with coordinates > mean 
            mean_centers = self.xp.mean(centers, axis=0)
            idx_high = centers > mean_centers[self.xp.newaxis, :]
            new_centers[idx_high] -= distance

            # Average coordinates
            new_centers[:, 0] = self.xp.mean(new_centers[:, 0])
            new_centers[:, 1] = self.xp.mean(new_centers[:, 1])

            # Restore high coordinates
            new_centers[idx_high] += distance

        else:
            # Alternative logic when not averaging
            if self.do_pup_inter_or_union != 'None':
                coords_round = self.xp.round(new_centers)
                coords_decimal = new_centers - coords_round

                diff_remainder_x = float(self.xp.max(coords_decimal[:, 0]) - self.xp.min(coords_decimal[:, 0]))
                diff_remainder_y = float(self.xp.max(coords_decimal[:, 1]) - self.xp.min(coords_decimal[:, 1]))

                new_centers[:, 0] = coords_round[:, 0] + self.xp.mean(coords_decimal[:, 0])
                new_centers[:, 1] = coords_round[:, 1] + self.xp.mean(coords_decimal[:, 1])

                delta_r = self.xp.sqrt(diff_remainder_x**2 + diff_remainder_y**2)

                if self.do_pup_inter_or_union == 'Union':
                    new_radii += delta_r
                elif self.do_pup_inter_or_union == 'Inter':
                    new_radii -= delta_r

        return new_radii, new_centers

    def _generate_pupil_indices(self, radii, centers, image_shape):
        """Equivalent to pyr_generate_index IDL function"""
        h, w = image_shape
        n_subaps = len([idx for idx in range(len(radii)) if radii[idx] > 0])

        if n_subaps == 0:
            raise ValueError("No valid pupils found")

        # Create coordinate grids
        y_coords, x_coords = self.xp.mgrid[0:h, 0:w]

        # Generate indices for each pupil
        max_radius = float(self.xp.max(radii))
        max_pixels_per_pupil = int(self.xp.pi * max_radius**2) + 100  # Safety margin
        ind_pup = self.xp.zeros((n_subaps, max_pixels_per_pupil), dtype=int)

        valid_pupil = 0
        for i in range(4):
            if radii[i] > 0:
                # Distance from pupil center
                dx = x_coords - centers[i, 0]
                dy = y_coords - centers[i, 1]
                distance = self.xp.sqrt(dx**2 + dy**2)

                # Pixels inside pupil
                inside_pupil = distance <= radii[i]
                pupil_indices = self.xp.where(inside_pupil)

                # Convert to flat indices
                flat_indices = self.xp.ravel_multi_index(pupil_indices, image_shape)
                n_pixels = len(flat_indices)

                if n_pixels > max_pixels_per_pupil:
                    raise ValueError(f"Pupil {i} has too many pixels: {n_pixels}")

                # Store indices
                ind_pup[valid_pupil, :n_pixels] = flat_indices
                if n_pixels < max_pixels_per_pupil:
                    ind_pup[valid_pupil, n_pixels:] = flat_indices[0]  # Pad with first index

                valid_pupil += 1

        # Resize to actual number of pixels used
        actual_max_pixels = int(self.xp.max([self.xp.sum(ind_pup[i, :] != ind_pup[i, 0]) + 1 
                                        for i in range(n_subaps)]))
        ind_pup_final = ind_pup[:, :actual_max_pixels]

        return ind_pup_final

    def finalize(self):
        """Save pupil data to file"""
        if self.pupdata is None:
            raise ValueError("No pupil data to save - trigger_code() may have failed")

        filename = self._filename
        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self._data_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.pupdata.save(file_path)

        if self.verbose:
            print(f'Saved pupil data to: {file_path}')
            print(f'Number of subapertures: {self.pupdata.n_subap}')