from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.lib.compute_zern_ifunc import compute_zern_ifunc

import logging
import warnings
import numpy as np

class ModalAnalysis(BaseProcessingObj):
    """
    Modal analysis processing object. Decomposition of an
    input ElectricField into modes defined
    by an influence function (e.g. Zernike polynomials).
    """
    def __init__(self,
                ifunc: IFunc=None,
                ifunc_inv: IFuncInv=None,
                type_str: str=None,
                npixels: int=None,
                obsratio: float=None,
                diaratio: float=None,
                pupilstop: Pupilstop=None,
                nmodes: int=None,
                wavelengthInNm: float=0.0,
                dorms: bool=None,
                n_inputs: int=1,
                remove_piston: bool=True,
                target_device_idx: int=None,
                precision: int=None):
        """
        Parameters
        ----------
        ifunc : IFunc, optional
            Influence function object defining the modes (default: None)
        ifunc_inv : IFuncInv, optional
            Inverse influence function object (default: None).
            If both ifunc and ifunc_inv are provided, ifunc_inv will be used.
        type_str : str, optional
            Type of influence function to compute if ifunc is not provided (e.g. 'zernike')
            (default: None)
        npixels : int, optional
            Number of pixels across the pupil (required if ifunc is not provided)
        obsratio : float, optional
            Obscuration ratio for influence function computation (required if ifunc is not provided)
        diaratio : float, optional
            Diameter ratio for influence function computation (required if ifunc is not provided)
        pupilstop : Pupilstop, optional
            Pupil stop object defining the mask to apply to the influence functions
            (default: None)
        nmodes : int, optional
            Number of modes to compute (default: None, meaning all modes)
        wavelengthInNm : float, optional
            Wavelength in nanometers for phase to mode conversion
            (default: 0.0, meaning no conversion)
        dorms : bool, optional
            Deprecated and ignored: the RMS of the wavefront is always computed.
            Kept for backward compatibility, a FutureWarning is issued if set.
        n_inputs : int, optional
            Number of input electric fields to process (default: 1).
            If greater than 1, the in_ef_list input will be used instead of in_ef.
        remove_piston : bool, optional
            Whether to remove the global piston term from the modes when inverting
            the influence function (default: True)
        target_device_idx : int [1], optional
            Target device for computation (-1 for CPU, >=0 for GPU)
        precision : int [1], optional
            Numerical precision (0 for double, 1 for single)
        """

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        mask = None
        if pupilstop:
            mask = pupilstop.A

        if ifunc_inv is not None:
            # ifunc_inv has priority over ifunc
            if nmodes is not None and nmodes != ifunc_inv.nmodes():
                ifunc_inv = IFuncInv(ifunc_inv.ifunc_inv[:, :nmodes],
                                     mask=ifunc_inv.mask_inf_func,
                                     target_device_idx=ifunc_inv.target_device_idx,
                                     precision=ifunc_inv.precision)
            self.phase2modes = ifunc_inv
        elif ifunc is not None:
            self.phase2modes = ifunc.inverse(nmodes=nmodes, remove_piston=remove_piston)
        else:
            if type_str is None:
                raise ValueError('At least one of ifunc and type must be set')
            if mask is not None:
                mask = (self.to_xp(mask) > 0).astype(self.dtype)
            if npixels is None:
                raise ValueError("If ifunc is not set, then npixels must be set!")

            type_lower = type_str.lower()
            if type_lower in ['zern', 'zernike']:
                ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=obsratio,
                                                 diaratio=diaratio, mask=mask,
                                                 xp=self.xp, dtype=self.dtype)
            else:
                raise ValueError(f'Invalid ifunc type {type_str}')

            ifunc = IFunc(ifunc, mask=mask, nmodes=nmodes, target_device_idx=self.target_device_idx)
            # compute_zern_ifunc already removes the piston from each mode
            self.phase2modes = ifunc.inverse(remove_piston=False)

        self._n_modes = self.phase2modes.nmodes()
        if nmodes is not None and nmodes > self._n_modes:
            raise ValueError(f'nmodes ({nmodes}) is larger than the number of available modes ({self._n_modes})')
        self._n_inputs = n_inputs

        if dorms is not None:
            warnings.warn('ModalAnalysis: dorms is deprecated and ignored, '
                          'the RMS is always computed', FutureWarning, stacklevel=2)
        self.wavelengthInNm = wavelengthInNm
        if wavelengthInNm > 0:
            mask_inf_func = self.phase2modes.mask_inf_func
            self._nm_to_rad_masked = mask_inf_func.astype(self.dtype) * (2 * np.pi / wavelengthInNm)
            # Eigenvalues of the finite difference Laplacian in the DCT domain
            rows, cols = mask_inf_func.shape
            v = self.xp.cos(np.pi * self.xp.arange(rows, dtype=self.dtype) / rows)
            u = self.xp.cos(np.pi * self.xp.arange(cols, dtype=self.dtype) / cols)
            self._laplacian_denom = 2 * (v.reshape(-1, 1) + u - 2)
            self._laplacian_denom[0, 0] = 1.0  # avoid division by zero

        # One row per input: all inputs are projected with a single matrix product,
        # reading the (possibly very large) inverse matrix only once.
        # The values of the list outputs are views on rows of these buffers.
        n_rows = max(n_inputs, 1)
        self._ph = self.xp.zeros((n_rows, self.phase2modes.npoints()), dtype=self.dtype)
        self._modes = self.xp.zeros((n_rows, self._n_modes), dtype=self.dtype)
        self._rms = self.xp.zeros(n_rows, dtype=self.dtype)

        # Single input outputs, only updated for in_ef
        self.out_modes = BaseValue('output modes from modal analysis',
                                   target_device_idx=target_device_idx,
                                   precision=precision)
        self.out_modes.value = self.xp.zeros(self._n_modes, dtype=self.dtype)
        self.rms = BaseValue('output RMS of phase from modal reconstructor',
                             target_device_idx=target_device_idx,
                             precision=precision)
        self.rms.value = self.xp.zeros(1, dtype=self.dtype)

        self.inputs['in_ef'] = InputValue(type=ElectricField, optional=True)
        self.inputs['in_ef_list'] = InputList(type=ElectricField, optional=True)
        self.outputs['out_modes'] = self.out_modes
        self.outputs['rms'] = self.rms
        self.out_modes_list = []
        self.rms_list = []
        for i in range(self._n_inputs):
            out_modes = BaseValue('modes', target_device_idx=self.target_device_idx,
                                  precision=precision)
            out_modes.value = self._modes[i]
            rms = BaseValue('phase RMS', target_device_idx=self.target_device_idx,
                            precision=precision)
            rms.value = self._rms[i:i+1]
            self.out_modes_list.append(out_modes)
            self.rms_list.append(rms)
            # Also available as a single output, e.g. for a DataStore
            self.outputs[f'out_modes_{i}'] = out_modes
        self.outputs['out_modes_list'] = self.out_modes_list
        self.outputs['rms_list'] = self.rms_list

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field for modal analysis (optional, use with in_ef_list)'),
                'in_ef_list': InputDesc(ElectricField, 'List of input electric fields for multi-source modal analysis (optional)')}

    @classmethod
    def output_names(cls):
        return {'out_modes': OutputDesc(BaseValue, 'Modal coefficients of the single input electric field (in_ef only, not updated with in_ef_list)'),
                'rms': OutputDesc(BaseValue, 'RMS of the wavefront of the single input electric field (in_ef only, not updated with in_ef_list)'),
                'out_modes_list': OutputDesc(list, 'Per-input modal coefficient vectors (list, one per connected input)'),
                'rms_list': OutputDesc(list, 'Per-input wavefront RMS (list, one per connected input)'),
                'out_modes_{input_idx}': OutputDesc(BaseValue, 'Modal coefficients of input [input_idx], same as out_modes_list[input_idx]')}

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.in_ef = self.local_inputs['in_ef']
        self.in_ef_list = self.local_inputs['in_ef_list']

    # Least squares phase unwrapping by solving the Poisson equation using the discrete cosine transform
    # Z. Zhao, Phase Unwrapping Algorithms by Solving the Poisson Equation
    def unwrap_ls(self, phase_wrap):

        # Wrapped phase differences (Gradients)
        dx = self.xp.diff(phase_wrap, axis=1)
        dx = self.xp.mod(dx + np.pi, 2 * np.pi) - np.pi

        dy = self.xp.diff(phase_wrap, axis=0)
        dy = self.xp.mod(dy + np.pi, 2 * np.pi) - np.pi

        # Calculate the Divergence (right-hand side of Poisson equation)
        rows, cols = phase_wrap.shape
        rho = self.xp.zeros((rows, cols), dtype=self.dtype)
        rho[:, 1:-1] = self.xp.diff(dx, axis=1)
        rho[1:-1, :] += self.xp.diff(dy, axis=0)

        # Boundary conditions
        rho[:, 0] = dx[:, 0]
        rho[:, -1] = -dx[:, -1]
        rho[0, :] += dy[0, :]
        rho[-1, :] += -dy[-1, :]

        # 2D discrete cosine transform
        dct_rho = self.dct(self.dct(rho, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Solve in frequency domain
        dct_phi = dct_rho / self._laplacian_denom
        dct_phi[0, 0] = 0.0 # avoid division by zero

        # Inverse 2D DCT
        return self.idct(self.idct(dct_phi, axis=0, norm='ortho'), axis=1, norm='ortho')

    def setup(self):
        super().setup()
        input_list = self.local_inputs['in_ef_list']
        if input_list:
            if self._n_inputs != len(input_list):
                raise ValueError(f"Number of inputs ({len(input_list)}) does not match expected number ({self._n_inputs})")

    def trigger_code(self):
        # Each input mode only updates its own outputs
        if self.in_ef:
            ef_list = [self.in_ef]
            modes, rms = self.out_modes.value[None, :], self.rms.value
            outputs = [self.out_modes, self.rms]
        else:
            ef_list = self.in_ef_list
            modes, rms = self._modes[:len(ef_list)], self._rms[:len(ef_list)]
            outputs = self.out_modes_list + self.rms_list

        idx = self.phase2modes.idx_inf_func
        ph = self._ph[:len(ef_list)]
        for i, current_ef in enumerate(ef_list):
            if self.wavelengthInNm > 0:
                phase_in_rad = self.unwrap_ls(current_ef.phaseInNm * self._nm_to_rad_masked)
                ph[i] = phase_in_rad[idx] * (self.wavelengthInNm / (2 * np.pi))
            else:
                ph[i] = current_ef.phaseInNm[idx]

        # modes and rms are views on the output values
        modes[:] = ph @ self.phase2modes.ifunc_inv
        rms[:] = self.xp.std(ph, axis=1)
        for output in outputs:
            output.generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()
        if self.logger.isEnabledFor(logging.DEBUG):
            if self.in_ef is not None:
                outputs = zip([self.out_modes], [self.rms])
            else:
                outputs = zip(self.out_modes_list, self.rms_list)
            for out, rms in outputs:
                self.logger.debug(f'First residual values: {out.value[:6]}')
                self.logger.debug(f'Phase RMS: {rms.value}')
