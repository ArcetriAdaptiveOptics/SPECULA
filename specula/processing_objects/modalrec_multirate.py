from specula import np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList
from specula.data_objects.slopes import Slopes

class ModalrecMultirate(BaseProcessingObj):
    """
    Multirate Tomographic Reconstructor processing object (for MORFEO-like systems).
    
    This object dynamically selects the appropriate Reconstruction Matrix (Recmat)
    based on which sensors have provided a new measurement at the current time step.
    It outputs a fixed-size vector of modes (e.g., 9 modes: 3x Tip-Tilt + 3x Plate Scale),
    where unobservable modes are naturally attenuated by the MMSE reconstructor.
    """

    def __init__(self,
                 recmat_dict: dict,
                 validity_masks: list,
                 n_modes_total: int,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters:
        -----------
        recmat_dict : dict
            A dictionary of Recmat objects loaded by SPECULA's YAML parser (using _dict_ref).
        validity_masks : list of lists
            List of boolean masks corresponding to the matrices in recmat_dict.
            Required to explicitly map objects to sensor validity states.
        n_modes_total : int
            The total size of the output modal vector (e.g., 9 for MORFEO LO loop).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if not recmat_dict:
            raise ValueError("recmat_dict cannot be empty.")
        if not validity_masks:
            raise ValueError("validity_masks must be provided to map reconstruction matrices to sensor states.")

        self.n_modes_total = n_modes_total
        self.recmat_dict = {}

        # =====================================================================
        # DICTIONARY MAPPING
        # =====================================================================
        rec_objects = list(recmat_dict.values())
        
        if len(rec_objects) != len(validity_masks):
            raise ValueError(f"Number of matrices ({len(rec_objects)}) and "
                             f"masks ({len(validity_masks)}) do not match.")
        
        for mask, rec_obj in zip(validity_masks, rec_objects):
            self.recmat_dict[tuple(mask)] = rec_obj

        # =====================================================================
        # SANITY CHECKS (Dimensions and Consistency)
        # =====================================================================
        if self.recmat_dict:
            n_sensors = len(list(self.recmat_dict.keys())[0])
            all_true_mask = tuple([True] * n_sensors)
            
            # Baseline maximum number of slopes (columns) from the all-True state
            max_cols = 0
            if all_true_mask in self.recmat_dict:
                max_cols = self.recmat_dict[all_true_mask].recmat.shape[1]

            for mask, rec_obj in self.recmat_dict.items():
                mat = rec_obj.recmat
                
                # Check A: Matrix row size must perfectly match n_modes_total
                if mat.shape[0] != self.n_modes_total:
                    raise ValueError(f"Matrix for mask {mask} has {mat.shape[0]} rows, "
                                     f"but n_modes_total is defined as {self.n_modes_total}.")
                
                # Check B: Matrix must accept at least some slopes
                if mat.shape[1] == 0:
                    raise ValueError(f"Matrix for mask {mask} has 0 columns. "
                                     f"It must accept at least some slopes.")

                # Check C: Dropping sensors should decrease or maintain the number of columns
                if max_cols > 0 and mat.shape[1] > max_cols:
                    raise ValueError(f"Logical inconsistency: mask {mask} requires {mat.shape[1]} slopes (columns), "
                                     f"which exceeds the baseline all-True state ({max_cols} slopes). "
                                     f"Dropping sensors cannot increase the number of input slopes!")

        # Prepare the output value
        self.out_modes = BaseValue('output dynamic modes from multirate reconstructor',
                                   target_device_idx=target_device_idx,
                                   precision=precision)

        # Initialize output with zeros
        self.out_modes.value = self.xp.zeros(self.n_modes_total, dtype=self.dtype)

        # Define Inputs and Outputs
        self.inputs['in_slopes_list'] = InputList(type=Slopes)
        self.outputs['out_modes'] = self.out_modes

        self._n_sensors = 0

    def setup(self):
        super().setup()

        slopes_list = self.local_inputs['in_slopes_list']
        if not slopes_list:
            raise ValueError("in_slopes_list must be connected.")

        self._n_sensors = len(slopes_list)

        # Move all matrices in the dictionary to the correct device (GPU/CPU)
        self.xp_recmat_dict = {}
        for validity_tuple, recmat_obj in self.recmat_dict.items():
            if len(validity_tuple) != self._n_sensors:
                raise ValueError(f"Validity tuple {validity_tuple} length does not match "
                                 f"number of sensors ({self._n_sensors}).")

            self.xp_recmat_dict[validity_tuple] = self.to_xp(recmat_obj.recmat, dtype=self.dtype)

    def trigger_code(self):
        slopes_list = self.local_inputs['in_slopes_list']

        validity = []
        valid_slopes = []

        # 1. Check which sensors have fresh data (Dynamic Scheduler)
        for s in slopes_list:
            is_valid = s.generation_time == self.current_time
            validity.append(is_valid)

            if is_valid:
                valid_slopes.append(s.slopes)

        validity_tuple = tuple(validity)

        # 2. If no sensors are valid, output zeros (Zero-Stuffing natural behavior)
        if not any(validity):
            self.out_modes.value[:] = 0.0
            self.out_modes.generation_time = self.current_time
            return

        # 3. Fetch the correct matrix from the pre-computed Look-Up Table
        if validity_tuple not in self.xp_recmat_dict:
            raise KeyError(f"No reconstruction matrix provided for validity state {validity_tuple}")

        current_recmat = self.xp_recmat_dict[validity_tuple]

        # 4. Concatenate all valid slopes into a single vector s_avail[k]
        s_avail = self.xp.hstack(valid_slopes)

        # 5. Matrix-Vector Multiplication: m[k] = R_v[k] * s_avail[k]
        output_modes = current_recmat @ s_avail

        # 6. Assign to output and update generation time
        self.out_modes.value = output_modes
        self.out_modes.generation_time = self.current_time
