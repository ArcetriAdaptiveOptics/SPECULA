from typing import List

from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.connections import InputValue
from iteras.base_value import BaseValue
from iteras.data_objects.matrix import Matrix
from iteras.lib.graph_matmul import matmul

class SoftLimiter(BaseProcessingObj):
    """
    Soft Limiter processing object.

    A generalized temporal regularizer for unobservable or poorly sensed 
    modes, based on a Minimum Mean Square Error (MMSE) subspace estimation. 
    It acts as a localized leaky integrator to selectively prevent the 
    control-induced divergence of specific modal subspaces without affecting 
    the rest of the control loop.

    The correction applied to the input command is
    ``gain * reprojector @ (estimator @ in_comm)``, where the estimator maps
    the command to the coordinates of the subspace (for example an MMSE
    estimator built from prior statistics) and the reprojector maps them
    back to the command space. It is applied at every step from
    ``start_time``, or every ``interval_time`` seconds.

    Typical application: draining from the accumulated command the modes
    that the sensors cannot see, which would otherwise drift, leaving the
    rest of the correction intact.
    """
    def __init__(self,
                 matrix_list: List[Matrix] = None,
                 gain: float = 1.0,
                 start_time: float = 0.0,
                 interval_time: float = 0.0,
                 recmat_list: List[Matrix] = None,
                 target_device_idx=None,
                 precision=None):
        """
        Parameters
        ----------
        matrix_list : list of Matrix
            The estimator, of shape (subspace size, command size), and the
            reprojector, of shape (command size, subspace size).
        recmat_list : list of Matrix
            Previous name of *matrix_list*, kept for compatibility
            (``recmat_list_object`` in the parameter files).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if matrix_list is not None and recmat_list is not None:
            raise ValueError('Only one of matrix_list and recmat_list can be given')
        if matrix_list is None:
            matrix_list = recmat_list
        if matrix_list is None or len(matrix_list) < 2:
            raise ValueError("SoftLimiter requires 'matrix_list' to contain at least two objects:"
                             " [estimator, reprojector].")

        self.gain = gain

        # Discrete time management
        self.start_time_t = self.seconds_to_t(start_time)
        self.interval_time_t = self.seconds_to_t(interval_time)
        self.last_correction_time = -1

        self.inputs['in_comm'] = InputValue(type=BaseValue)

        self.outputs['out_comm'] = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=self.precision)
        self.outputs['out_ost'] = BaseValue(target_device_idx=self.target_device_idx,
                                            precision=self.precision)

        self.estimator = self.to_xp(matrix_list[0].matrix, dtype=self.dtype)    # (n_sub, n_modes)
        self.reprojector = self.to_xp(matrix_list[1].matrix, dtype=self.dtype)  # (n_modes, n_sub)

        if self.estimator.ndim != 2 or self.reprojector.ndim != 2:
            raise ValueError("SoftLimiter expects 2D matrices for both the estimator and the reprojector.")

        self.n_sub, self.n_modes = self.estimator.shape
        if self.reprojector.shape != (self.n_modes, self.n_sub):
            raise ValueError(
                "Incompatible matrix shapes for SoftLimiter: "
                f"estimator shape {tuple(self.estimator.shape)} and reprojector shape "
                f"{tuple(self.reprojector.shape)}. "
                f"Expected reprojector shape ({self.n_modes}, {self.n_sub})."
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
            'out_comm': OutputDesc(BaseValue, 'Cleaned command vector'),
            'out_ost': OutputDesc(BaseValue, 'State correction vector, for example for an IirFilter')
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
            # 1. Estimation: coordinates of the command in the subspace
            est_sub = matmul(self.estimator, in_comm, self.xp)

            # 2. Re-projection: back into the command space
            delta_comm_full = matmul(self.reprojector, est_sub, self.xp)

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
