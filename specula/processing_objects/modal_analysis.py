from specula import fuse, show_in_profiler

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.ef import ElectricField
from specula.data_objects.ifunc import IFunc

class ModalAnalysis(BaseProcessingObj):

    def __init__(self, 
                ifunc: IFunc = None,
                wavelengthInNm: float = 0,
                dorms: bool = False,
                target_device_idx: int = None,
                precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.ifunc = ifunc
        self.phase2modes = self.xp.asanyarray(ifunc.inverse(), dtype=self.dtype)
        self.rms = BaseValue('modes', 'output RMS of modes from modal reconstructor')        
        self.dorms = dorms
        self.wavelengthInNm = wavelengthInNm
        self.verbose = False  # Verbose flag for debugging output
        self.out_modes = BaseValue('output modes from modal analysis', target_device_idx=target_device_idx)        
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_modes'] = self.out_modes


    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.in_ef = self.local_inputs['in_ef']

    def unwrap_2d(self, p):
        unwrapped_p = self.xp.copy(p)
        for r in range(p.shape[1]):
            row = unwrapped_p[:, r]
            unwrapped_p[:, r] = self.xp.unwrap(row)        
        for c in range(p.shape[0]):
            col = unwrapped_p[c, :]
            unwrapped_p[c, :] = self.xp.unwrap(col)        
        return unwrapped_p


    def trigger_code(self):
        if self.ifunc.zeroPad:
            m = self.xp.dot(self.in_ef.phaseInNm, self.phase2modes)
        else:
            if self.wavelengthInNm > 0:
                phase_in_rad = self.in_ef.phaseInNm * (2 * self.xp.pi / self.wavelengthInNm)
                phase_in_rad *= self.ifunc.mask_inf_func.astype(float)
                phase_in_rad = self.unwrap_2d(phase_in_rad)
                phase_in_nm = phase_in_rad * (self.wavelengthInNm / (2 * self.xp.pi))
                ph = phase_in_nm[self.ifunc.idx_inf_func]
            else:
                ph = self.in_ef.phaseInNm[self.ifunc.idx_inf_func]

            m = self.xp.dot(ph, self.phase2modes)

        self.out_modes.value = m
        self.out_modes.generation_time = self.current_time

        if self.dorms:
            self.rms.value = self.xp.std(ph)
            self.rms.generation_time = self.current_time

        if self.verbose:
            print(f"First residual values: {m[:min(6, len(m))]}")
            if self.dorms:
                print(f"Phase RMS: {self.rms.value}")

    # TODO
    def run_check(self, time_step):
        errmsg = ""
        #if self.in_ef is None:
        #    errmsg += "EF is not valid. "
        #if self.ifunc is None:
        #    errmsg += "ifunc is not valid. "
        return 1

