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
                 recmat_dict: dict[str, Recmat],
                 validity_masks: list | None,
                 n_modes_total: int,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters:
        -----------
        recmat_dict : dict
            A dictionary of Recmat objects loaded by SPECULA's YAML parser (using _dict_ref).
        validity_masks : list of lists, optional
            List of boolean masks corresponding to the matrices in recmat_dict.
            If None, masks are inferred from recmat_dict keys when possible.
        n_modes_total : int
            The total size of the output modal vector (M).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if not recmat_dict:
            raise ValueError("recmat_dict cannot be empty.")

        self.n_modes_total = n_modes_total
        self.recmat_dict = {}
        self.xp_recmat_dict = {}

        # =====================================================================
        # DICTIONARY MAPPING
        # =====================================================================
        if validity_masks is None:
            for key, rec_obj in recmat_dict.items():
                if isinstance(key, tuple):
                    mask_tuple = key
                elif isinstance(key, list):
                    mask_tuple = tuple(key)
                elif isinstance(key, str):
                    bitstring = key
                    if '_v' in key:
                        bitstring = key.rsplit('_v', 1)[1]
                    if not bitstring or any(ch not in '01' for ch in bitstring):
                        raise ValueError("Cannot infer validity mask from recmat_dict key "
                                         f"'{key}'. Use tuple/list keys, '_v<bits>' suffix, "
                                         "or provide validity_masks explicitly.")
                    mask_tuple = tuple(ch == '1' for ch in bitstring)
                else:
                    raise ValueError("Cannot infer validity mask from non-string/non-sequence key "
                                     f"'{key}'.")

                if mask_tuple in self.recmat_dict:
                    raise ValueError(f"Duplicated validity mask {mask_tuple}.")
                self.recmat_dict[mask_tuple] = rec_obj
        else:
            rec_objects = list(recmat_dict.values())
            if len(rec_objects) != len(validity_masks):
                raise ValueError(f"Number of matrices ({len(rec_objects)}) and "
                                 f"masks ({len(validity_masks)}) do not match.")

            for mask, rec_obj in zip(validity_masks, rec_objects):
                mask_tuple = tuple(mask)
                if mask_tuple in self.recmat_dict:
                    raise ValueError(f"Duplicated validity mask {mask_tuple}.")
                self.recmat_dict[mask_tuple] = rec_obj

        # =====================================================================
        # SANITY CHECKS (Dimensions)
        # =====================================================================
        if self.recmat_dict:
            for mask, rec_obj in self.recmat_dict.items():
                mat = rec_obj.recmat
                if mat.shape[0] != self.n_modes_total:
                    raise ValueError(f"Matrix for mask {mask} has {mat.shape[0]} rows, "
                                     f"but n_modes_total is defined as {self.n_modes_total}.")

        # Infer number of sensors from validity tuples and create outputs upfront.
        mask_lengths = {len(mask) for mask in self.recmat_dict.keys()}
        if len(mask_lengths) != 1:
            raise ValueError("All validity masks must have the same length.")

        self.n_sensors = next(iter(mask_lengths))
        if self.n_sensors < 1:
            raise ValueError("At least one sensor is required.")

        # Register input port
        self.inputs['in_slopes_list'] = InputList(type=Slopes)

        # Create fixed output ports using topology inferred from mask tuples.
        self.out_modes_list = []
        for i in range(self.n_sensors):
            out_val = BaseValue(f'output modes for sensor {i}',
                                target_device_idx=self.target_device_idx,
                                precision=self.precision)
            out_val.value = self.xp.zeros(self.n_modes_total, dtype=self.dtype)

            port_name = f'out_modes_{i}'
            self.outputs[port_name] = out_val
            self.out_modes_list.append(out_val)

    def setup(self):
        super().setup()

        slopes_list = self.local_inputs['in_slopes_list']
        if not slopes_list:
            raise ValueError("in_slopes_list must be connected.")

        if len(slopes_list) != self.n_sensors:
            raise ValueError(f"Connected sensors ({len(slopes_list)}) do not match "
                             f"reconstructor topology ({self.n_sensors}).")

        # Move matrices to the target device
        slopes_per_sensor = [s.slopes.shape[0] for s in slopes_list]
        for validity_tuple, recmat_obj in self.recmat_dict.items():
            if len(validity_tuple) != self.n_sensors:
                raise ValueError(f"Validity tuple {validity_tuple} length does not match "
                                 f"number of connected sensors ({self.n_sensors}).")

            expected_cols = sum(slopes_per_sensor[i] for i, active in enumerate(validity_tuple) if active)
            n_cols = recmat_obj.recmat.shape[1]
            if n_cols != expected_cols:
                raise ValueError(f"Matrix for mask {validity_tuple} has {n_cols} columns, "
                                 f"expected {expected_cols} from active sensors.")

            self.xp_recmat_dict[validity_tuple] = self.to_xp(recmat_obj.recmat, dtype=self.dtype)

    def trigger_code(self):
        slopes_list = self.local_inputs['in_slopes_list']

        validity = []
        for s in slopes_list:
            validity.append(s.generation_time == self.current_time)

        validity_tuple = tuple(validity)

        # 1. Zero-Stuffing condition: No sensors active
        if not any(validity):
            for i in range(self.n_sensors):
                self.out_modes_list[i].value[:] = 0.0
                self.out_modes_list[i].generation_time = self.current_time
            return

        # 2. Fetch the correct matrix from the Look-Up Table
        if validity_tuple not in self.xp_recmat_dict:
            raise KeyError(f"No reconstruction matrix provided for validity state {validity_tuple}")

        current_recmat = self.xp_recmat_dict[validity_tuple]

        # 3. Dynamic Matrix Slicing and Multiplication
        col_offset = 0
        for i, s in enumerate(slopes_list):
            if validity[i]:
                n_slopes = s.slopes.shape[0]

                # Extract the M x (N_slopes) block for this specific sensor
                R_block = current_recmat[:, col_offset : col_offset + n_slopes]

                # Project this sensor slopes into the full M-dimensional modal space
                self.out_modes_list[i].value = R_block @ s.slopes
                col_offset += n_slopes
            else:
                # Sensor is inactive, output M zeros
                self.out_modes_list[i].value[:] = 0.0

            self.out_modes_list[i].generation_time = self.current_time
