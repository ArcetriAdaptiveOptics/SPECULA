from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.lib.compute_zern_ifunc import compute_zern_ifunc

class ModalAnalysis(BaseProcessingObj):

    def __init__(self,
                ifunc: IFunc=None,
                ifunc_inv: IFuncInv=None,
                type_str: str=None,
                npixels: int=None,
                nzern: int=None,            # TODO not used
                obsratio: float=None,
                diaratio: float=None,
                pupilstop: Pupilstop=None,
                nmodes: int=None,
                wavelengthInNm: float=0.0,
                dorms: bool=False,
                n_inputs: int=1,
                target_device_idx: int=None,
                precision: int=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        mask = None
        if pupilstop:
            mask = pupilstop.A

        if ifunc is None and ifunc_inv is None:
            if type_str is None:
                raise ValueError('At least one of ifunc and type must be set')
            if mask is not None:
                mask = (self.to_xp(mask) > 0).astype(self.dtype)
            if npixels is None:
                raise ValueError("If ifunc is not set, then npixels must be set!")

            type_lower = type_str.lower()
            if type_lower in ['zern', 'zernike']:
                ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                                 xp=self.xp, dtype=self.dtype)
            else:
                raise ValueError(f'Invalid ifunc type {type_str}')

            ifunc = IFunc(ifunc, mask=mask, nmodes=nmodes, target_device_idx=self.target_device_idx)
            self.phase2modes = ifunc.inverse()
        elif ifunc is None and ifunc_inv is not None:
            # Use ifunc_inv directly, don't attempt to call inverse() on None
            self.phase2modes = ifunc_inv
        elif ifunc is not None and ifunc_inv is None:
            # This is the case where only ifunc is provided
            if nmodes is not None and nmodes != ifunc.size[0]:
                ifunc.cut(nmodes=nmodes)
            self.phase2modes = ifunc.inverse()
        else:  # Both are provided
            # Prioritize ifunc_inv
            self.phase2modes = ifunc_inv

        self.rms = BaseValue('output RMS of phase from modal reconstructor')
        self.rms.value = self.xp.zeros(1, dtype=self.dtype)
        self.dorms = dorms
        self.wavelengthInNm = wavelengthInNm
        self.verbose = False  # Verbose flag for debugging output

        if nmodes is None:
            self._n_modes = self.phase2modes.size[1]
        else:
            self._n_modes = nmodes
        self._n_inputs = n_inputs

        self.out_modes = BaseValue('output modes from modal analysis', target_device_idx=target_device_idx)
        self.out_modes.value = self.xp.zeros(self._n_modes, dtype=self.dtype)
        self.inputs['in_ef'] = InputValue(type=ElectricField, optional=True)
        self.inputs['in_ef_list'] = InputList(type=ElectricField, optional=True)
        self.outputs['out_modes'] = self.out_modes
        self.outputs['rms'] = self.rms
        self.outputs['out_modes_list'] = []
        for _ in range(self._n_inputs):
            self.outputs['out_modes_list'].append(BaseValue('modes', target_device_idx=self.target_device_idx))
        self.out_modes_list = self.outputs['out_modes_list']

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

    def setup(self):
        super().setup()
        input_list = self.local_inputs['in_ef_list']
        if input_list:
            if self._n_inputs != len(input_list):
                raise ValueError(f"Number of inputs ({len(input_list)}) does not match expected number ({self._n_inputs})")
            for i in range(len(input_list)):
                self.outputs['out_modes_list'][i].value = self.xp.zeros(self._n_modes, dtype=self.dtype)

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

        self.out_modes.value[:] = m
        self.out_modes.generation_time = self.current_time

        for li, current_ef in enumerate(ef_list):
            if self.phase2modes._doZeroPad:
                m = self.xp.dot(current_ef.phaseInNm, self.phase2modes.ifunc_inv)
            else:
                if self.wavelengthInNm > 0:
                    phase_in_rad = current_ef.phaseInNm * (2 * self.xp.pi / self.wavelengthInNm)
                    phase_in_rad *= self.phase2modes.mask_inf_func.astype(float)
                    phase_in_rad = self.unwrap_2d(phase_in_rad)
                    phase_in_nm = phase_in_rad * (self.wavelengthInNm / (2 * self.xp.pi))
                    ph = phase_in_nm[self.phase2modes.idx_inf_func]
                else:
                    ph = current_ef.phaseInNm[self.phase2modes.idx_inf_func]

                m = self.xp.dot(ph, self.phase2modes.ifunc_inv)

            output_list[li].value[:] = m
            output_list[li].generation_time = self.current_time

        if self.dorms:
            self.rms.value[:] = self.xp.std(ph)
            self.rms.generation_time = self.current_time

        if self.verbose:
            print(f"First residual values: {m[:min(6, len(m))]}")
            if self.dorms:
                print(f"Phase RMS: {self.rms.value}")
