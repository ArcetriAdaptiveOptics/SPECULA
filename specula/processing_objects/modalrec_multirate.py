from typing import List

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes

class ModalrecMultirate(BaseProcessingObj):
    """
    Multirate Tomographic Reconstructor processing object (for MORFEO-like systems).

    This object dynamically selects the appropriate Reconstruction Matrix (Recmat)
    based on which sensors have provided a new measurement at the current time step.

    It mathematically slices the selected matrix into N blocks (one per sensor),
    outputting N independent modal vectors of size M. The downstream multirate
    controller will fuse these partial modal projections.
    """

    def __init__(self,
                 recmat_list: List[Recmat],
                 validity_masks: List[List[bool]],
                 n_modes_total: int,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters
        ----------
        recmat_list : list of Recmat
            Ordered list of reconstruction matrices, one for each validity state.
            Matrices are expected to use the full sensor-vector geometry.
        validity_masks : list of list of bool
            Boolean masks associated with each matrix in `recmat_list`, using the
            same ordering.
        n_modes_total : int
            Total size of each output modal vector.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if n_modes_total is None:
            raise ValueError("n_modes_total must be provided.")
        if not recmat_list:
            raise ValueError("recmat_list cannot be empty.")
        if not validity_masks:
            raise ValueError("validity_masks cannot be empty.")
        if len(recmat_list) != len(validity_masks):
            raise ValueError(f"Number of matrices ({len(recmat_list)}) and masks"
                             f" ({len(validity_masks)}) do not match.")

        self.n_modes_total = n_modes_total
        self.recmat_by_mask = {}
        self.xp_recmat_by_mask = {}
        self.sensor_col_offsets = None

        # build a look-up table (dictionary) that maps each validity mask to
        # its corresponding reconstruction matrix
        for mask, rec_obj in zip(validity_masks, recmat_list):
            mask_tuple = tuple(mask)
            if mask_tuple in self.recmat_by_mask:
                raise ValueError(f"Duplicated validity mask {mask_tuple}.")
            self.recmat_by_mask[mask_tuple] = rec_obj

        # =====================================================================
        # SANITY CHECKS (Dimensions)
        # =====================================================================
        for mask, rec_obj in self.recmat_by_mask.items():
            mat = rec_obj.recmat
            if mat.shape[0] != self.n_modes_total:
                raise ValueError(f"Matrix for mask {mask} has {mat.shape[0]} rows, "
                                 f"but n_modes_total is defined as {self.n_modes_total}.")

        # Build a set of unique mask lengths to ensure all masks have the same number of sensors
        mask_lengths = {len(mask) for mask in self.recmat_by_mask.keys()}
        if len(mask_lengths) != 1:
            raise ValueError("All validity masks must have the same length.")

        # Convert the single element set to an iterable and extract
        # the first (and only) element to get the number of sensors
        self.n_sensors = next(iter(mask_lengths))
        if self.n_sensors < 1:
            raise ValueError("At least one sensor is required.")

        # Register input list
        self.inputs['in_slopes_list'] = InputList(type=Slopes)

        # Create fixed output list
        self.out_modes_list = []
        for i in range(self.n_sensors):
            out_val = BaseValue(f'output modes for sensor {i}',
                                target_device_idx=self.target_device_idx,
                                precision=self.precision)
            out_val.value = self.xp.zeros(self.n_modes_total, dtype=self.dtype)

            output_name = f'out_modes_{i}'
            self.outputs[output_name] = out_val
            self.out_modes_list.append(out_val)

    def setup(self):
        super().setup()

        slopes_list = self.local_inputs['in_slopes_list']

        if len(slopes_list) != self.n_sensors:
            raise ValueError(f"Connected sensors ({len(slopes_list)}) do not match "
                             f"reconstructor topology ({self.n_sensors}).")

        slopes_per_sensor = [s.slopes.shape[0] for s in slopes_list]
        total_cols = sum(slopes_per_sensor)
        self.sensor_col_offsets = [0]
        for n_slopes in slopes_per_sensor:
            self.sensor_col_offsets.append(self.sensor_col_offsets[-1] + n_slopes)

        for validity_tuple, recmat_obj in self.recmat_by_mask.items():
            if len(validity_tuple) != self.n_sensors:
                raise ValueError(f"Validity tuple {validity_tuple} length does not match "
                                 f"number of connected sensors ({self.n_sensors}).")

            n_cols = recmat_obj.recmat.shape[1]
            if n_cols != total_cols:
                raise ValueError(f"Matrix for mask {validity_tuple} has {n_cols} columns, "
                                 f"expected {total_cols} from the full sensor vector.")

            self.xp_recmat_by_mask[validity_tuple] = self.to_xp(recmat_obj.recmat,
                                                                dtype=self.dtype)

    def trigger_code(self):
        slopes_list = self.local_inputs['in_slopes_list']

        # Determine a list of booleans indicating which sensors have new data
        # at the current time step
        validity = []
        for s in slopes_list:
            validity.append(s.generation_time == self.current_time)

        validity_tuple = tuple(validity)

        # 1. Zero-Stuffing condition: No sensors active
        if not any(validity):
            for i in range(self.n_sensors):
                self.out_modes_list[i].value[:] = 0
                self.out_modes_list[i].generation_time = self.current_time
            return

        # 2. Fetch the correct matrix from the Look-Up Table
        if validity_tuple not in self.xp_recmat_by_mask:
            raise KeyError(f"No reconstruction matrix provided for validity state {validity_tuple}")

        current_recmat = self.xp_recmat_by_mask[validity_tuple]

        # 3. Dynamic Matrix Slicing and Multiplication
        for i, s in enumerate(slopes_list):
            if validity[i]:
                start = self.sensor_col_offsets[i]
                end = self.sensor_col_offsets[i + 1]

                # Extract the M x (N_slopes) block for this specific sensor
                R_block = current_recmat[:, start:end]

                # Project this sensor slopes into the full M-dimensional modal space
                self.out_modes_list[i].value[:] = R_block @ s.slopes
            else:
                # Sensor is inactive, output M zeros
                self.out_modes_list[i].value[:] = 0

            self.out_modes_list[i].generation_time = self.current_time
