from typing import List

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.recmat import Recmat

class SoftLimiter(BaseProcessingObj):
    """
    Soft Limiter processing object.

    A generalized temporal regularizer for unobservable or poorly sensed 
    modes, based on a Minimum Mean Square Error (MMSE) subspace estimation. 
    It acts as a localized leaky integrator to selectively prevent the 
    control-induced divergence of specific modal subspaces without affecting 
    the rest of the control loop.

    Primary application:
    Specifically configured to mitigate the 'Island Effect' in segmented 
    SCAO systems. It continuously drains unobservable differential pistons 
    (petal modes) from the accumulated command state using von Kármán 
    prior statistics, leaving the continuous atmospheric correction intact.
    """
    def __init__(self,
                 recmat_list: List[Recmat],
                 gain: float = 1.0,
                 start_time: float = 0.0,
                 interval_time: float = 0.0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if len(recmat_list) < 2:
            raise ValueError("SoftLimiter requires 'recmat_list' to contain at least two objects:"
                             " [MMSE_recmat, Intmat].")

        self.gain = gain

        # SPECULA discrete time management
        self.start_time_t = self.seconds_to_t(start_time)
        self.interval_time_t = self.seconds_to_t(interval_time)
        self.last_correction_time = -1

        self.inputs['in_comm'] = InputValue(type=BaseValue)

        self.outputs['out_comm'] = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=self.precision)
        self.outputs['out_ost'] = BaseValue(target_device_idx=self.target_device_idx,
                                            precision=self.precision)

        # Expect recmat_list[0] to be the MMSE reconstructor (N_pistons x N_modes) and
        #        recmat_list[1] to be the Interaction matrix (N_modes x N_pistons)
        self.logger.info("Loading MMSE reconstructor and Interaction matrices from recmat_list...")
        recmat_obj = recmat_list[0]
        intmat_obj = recmat_list[1]

        self.recmat = self.to_xp(recmat_obj.recmat, dtype=self.dtype) # Shape: (N_pistons_rel, N_modes)
        self.intmat = self.to_xp(intmat_obj.recmat, dtype=self.dtype) # Shape: (N_modes, N_pistons)

        if self.recmat.ndim != 2 or self.intmat.ndim != 2:
            raise ValueError("SoftLimiter expects 2D matrices for both recmat and intmat.")

        self.n_pistons, self.n_modes = self.recmat.shape
        intmat_n_modes, intmat_n_pistons = self.intmat.shape
        if (intmat_n_modes, intmat_n_pistons) != (self.n_modes, self.n_pistons):
            raise ValueError(
                "Incompatible matrix shapes for SoftLimiter: "
                f"recmat shape {tuple(self.recmat.shape)} and intmat shape {tuple(self.intmat.shape)}. "
                f"Expected intmat shape ({self.n_modes}, {self.n_pistons})."
            )

        self.logger.info(f"SoftLimiter initialized successfully. Gain: {self.gain}")

    def setup(self):
        super().setup()

        in_comm = self.local_inputs['in_comm'].value
        in_comm_size = int(self.xp.asarray(in_comm).size)
        if in_comm_size != self.n_modes:
            raise ValueError(
                f"Input command size ({in_comm_size}) does not match matrix mode size ({self.n_modes})."
            )

        self.outputs['out_comm'].set_value(self.xp.asarray(in_comm, dtype=self.dtype).copy())
        self.outputs['out_ost'].set_value(self.xp.zeros_like(in_comm, dtype=self.dtype))

    @classmethod
    def input_names(cls):
        return {'in_comm': InputDesc(BaseValue, 'Input command vector from integrators')}

    @classmethod
    def output_names(cls):
        return {
            'out_comm': OutputDesc(BaseValue, 'Cleaned command vector for the DM'),
            'out_ost': OutputDesc(BaseValue, 'State correction vector for the IirFilter')
        }

    def trigger_code(self):
        in_comm = self.local_inputs['in_comm'].value

        out_comm_val = in_comm.copy()
        out_ost_val = None

        do_check = False
        if self.current_time >= self.start_time_t:
            if self.interval_time_t <= 0:
                do_check = True
            elif (self.current_time - self.last_correction_time) >= self.interval_time_t:
                do_check = True

        if do_check:
            # 1. MMSE Estimation: Project full command into the 5 relative petal DOFs
            est_rel_petals = self.recmat @ in_comm

            # 2. Re-projection: Map the estimated 5 DOFs back into the full modal basis
            delta_comm_full = self.intmat @ est_rel_petals

            # 3. Apply the correction gain (Soft Limiting)
            delta_comm = delta_comm_full * self.gain

            # 4. Apply corrections
            out_comm_val -= delta_comm
            out_ost_val = delta_comm

            self.last_correction_time = self.current_time
        else:
            out_ost_val = self.xp.zeros_like(in_comm)

        self.outputs['out_comm'].set_value(out_comm_val)
        self.outputs['out_ost'].set_value(out_ost_val)

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_comm'].generation_time = self.current_time
        self.outputs['out_ost'].generation_time = self.current_time
