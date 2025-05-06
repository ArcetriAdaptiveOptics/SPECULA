from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.lib.compute_zern_ifunc import compute_zern_ifunc

class ModalAnalysis(BaseProcessingObj):

    def __init__(self, 
                ifunc: IFunc=None,
                ifunc_inv: IFuncInv=None,
                type_str: str=None,
                mask=None,
                npixels: int=None,
                nzern: int=None,
                obsratio: float=None,
                diaratio: float=None,
                nmodes: int=None,
                wavelengthInNm: float=0.0,
                dorms: bool=False,
                target_device_idx: int=None,
                precision: int=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if ifunc is None and ifunc_inv is None:
            if type_str is None:
                raise ValueError('At least one of ifunc and type must be set')
            if mask is not None:
                mask = (self.xp.array(mask) > 0).astype(self.dtype)
            if npixels is None:
                raise ValueError("If ifunc is not set, then npixels must be set!")
            
            type_lower = type_str.lower()
            if type_lower in ['zern', 'zernike']:
                ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                                 xp=self.xp, dtype=self.dtype)
            else:
                raise ValueError(f'Invalid ifunc type {type_str}')
            
            ifunc = IFunc(ifunc, mask=mask, nmodes=nmodes)
            self.phase2modes = ifunc.inverse()
        elif ifunc is None and ifunc_inv is not None:
            # Use ifunc_inv directly, don't attempt to call inverse() on None
            self.phase2modes = ifunc_inv
        elif ifunc is not None and ifunc_inv is None:
            # This is the case where only ifunc is provided
            self.phase2modes = ifunc.inverse()
        else:  # Both are provided
            # Prioritize ifunc_inv
            self.phase2modes = ifunc_inv

        self.rms = BaseValue('modes', 'output RMS of modes from modal reconstructor') 
        self.dorms = dorms
        self.wavelengthInNm = wavelengthInNm
        self.verbose = False  # Verbose flag for debugging output
        self.out_modes = BaseValue('output modes from modal analysis', target_device_idx=target_device_idx)
        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.inputs['in_ef_list'] = InputList(type=ElectricField)
        self.outputs['out_modes'] = self.out_modes
        self.outputs['rms'] = self.rms


    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.in_ef = self.local_inputs['in_ef']
        self.in_ef_list = self.local_inputs['in_ef_list']


    def unwrap_2d(self, p):
        unwrapped_p = self.xp.copy(p)
        for r in range(p.shape[1]):
            row = unwrapped_p[:, r]
            unwrapped_p[:, r] = self.xp.unwrap(row)        
        for c in range(p.shape[0]):
            col = unwrapped_p[c, :]
            unwrapped_p[c, :] = self.xp.unwrap(col)        
        return unwrapped_p

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)        
        for i in range(len(self.inputs['in_ef_list'].get())):
            self.outputs['out_modes_list'].append(BaseValue('modes', target_device_idx=self.target_device_idx))

        self.out_modes_list = self.outputs['out_modes_list']

    def trigger_code(self):
        if self.in_ef:
            ef_list = [self.in_ef]
            output_list = [self.out_modes]
        else:
            ef_list = self.in_ef_list
            output_list = self.out_modes_list
        if self.phase2modes._doZeroPad:
            m = self.xp.dot(self.in_ef.phaseInNm, self.phase2modes.ifunc_inv)
        else:
            if self.wavelengthInNm > 0:
                phase_in_rad = self.in_ef.phaseInNm * (2 * self.xp.pi / self.wavelengthInNm)
                phase_in_rad *= self.phase2modes.mask_inf_func.astype(float)
                phase_in_rad = self.unwrap_2d(phase_in_rad)
                phase_in_nm = phase_in_rad * (self.wavelengthInNm / (2 * self.xp.pi))
                ph = phase_in_nm[self.phase2modes.idx_inf_func]
            else:
                ph = self.in_ef.phaseInNm[self.phase2modes.idx_inf_func]

            m = self.xp.dot(ph, self.phase2modes.ifunc_inv)

        self.out_modes.value = m
        self.out_modes.generation_time = self.current_time

        for li, current_ef in enumerate(ef_list):
            if self.ifunc._doZeroPad:
                m = self.xp.dot(current_ef.phaseInNm, self.phase2modes)
            else:
                if self.wavelengthInNm > 0:
                    phase_in_rad = current_ef.phaseInNm * (2 * self.xp.pi / self.wavelengthInNm)
                    phase_in_rad *= self.ifunc.mask_inf_func.astype(float)
                    phase_in_rad = self.unwrap_2d(phase_in_rad)
                    phase_in_nm = phase_in_rad * (self.wavelengthInNm / (2 * self.xp.pi))
                    ph = phase_in_nm[self.ifunc.idx_inf_func]
                else:
                    ph = current_ef.phaseInNm[self.ifunc.idx_inf_func]

                m = self.xp.dot(ph, self.phase2modes)
            
            output_list[li].value = m
            output_list[li].generation_time = self.current_time
        
        if self.dorms:
            self.rms.value = self.xp.std(ph)
            self.rms.generation_time = self.current_time

        if self.verbose:
            print(f"First residual values: {m[:min(6, len(m))]}")
            if self.dorms:
                print(f"Phase RMS: {self.rms.value}")