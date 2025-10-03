from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dm import DM
from specula.lib.platescale_coeff import platescale_coeff

class LinearCombination(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 no_focus: bool = False,
                 no_lift: bool = False,
                 dm1: DM=None,
                 dm3: DM=None,
                 start_modes: list = [],
                 plate_scale_idx: int = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil

        self.no_focus = no_focus
        self.no_lift = no_lift
        self.plate_scale_idx = plate_scale_idx

        self.inputs['in_vectors_list'] = InputList(type=BaseValue)
        self.out_vector = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_vector'] = self.out_vector

        if dm1 is not None and dm3 is not None:
            # 0 because we looked at a single DM
            self.ps_coeff = self.xp.array(platescale_coeff([dm1,dm3], start_modes, self.pixel_pupil)[0])
        else:
            self.ps_coeff = self.xp.zeros(3)

    def trigger_code(self):
        in_vectors = self.local_inputs['in_vectors_list']
        idx = 0
        lgs = in_vectors[idx].value.copy()
        idx += 1
        if not self.no_focus:
            focus = in_vectors[idx].value
            idx += 1
        if not self.no_lift:
            lift = in_vectors[idx].value
            idx += 1
        ngs = in_vectors[idx].value

        # TIP / TILT
        lgs[0:2] = ngs[0:2]
        if not self.no_focus:
            lgs[2] = focus[0]
        if self.plate_scale_idx is not None:
            lgs[2:5] -= ngs[2:]*self.ps_coeff
            lgs[self.plate_scale_idx:self.plate_scale_idx+3] = ngs[2:]

        # LIFT
        if not self.no_lift:
            lgs = self.xp.concatenate([lgs, lift])

        self.out_vector.value *= 0.0
        self.out_vector.value[:len(lgs)] = lgs
        self.out_vector.generation_time = self.current_time

    def setup(self):
        super().setup()

        in_vectors = self.local_inputs['in_vectors_list']

        # Check input vectors consistency
        num_inputs = len(in_vectors)

        if num_inputs == 4:
            # LGS, FOCUS, LIFT, NGS - both must be False
            if self.no_focus or self.no_lift:
                raise ValueError("With 4 input vectors, both no_focus and no_lift must be False")
        elif num_inputs == 3:
            # Only one between FOCUS and LIFT is missing - only one must be True
            if self.no_focus == self.no_lift:
                raise ValueError("With 3 input vectors, exactly one of no_focus or no_lift must be True")
        elif num_inputs == 2:
            # Only LGS and NGS - both must be True
            if not self.no_focus or not self.no_lift:
                raise ValueError("With 2 input vectors, both no_focus and no_lift must be True")
        else:
            raise ValueError(f"Invalid number of input vectors: {num_inputs}. Expected 2, 3, or 4")

        idx = 0
        lgs = in_vectors[idx].value
        idx += 1
        if not self.no_focus:
            idx += 1
        if not self.no_lift:
            lift = in_vectors[idx].value
            idx += 1
        else:
            lift = self.xp.zeros(0,dtype=self.dtype)

        # Note: NGS and FOCUS override LGS values
        self.out_vector.value = self.xp.concatenate([lgs, lift]) * 0.0
