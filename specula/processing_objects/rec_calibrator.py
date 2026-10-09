import os

from specula.base_processing_obj import BaseProcessingObj, InputDesc
from specula.data_objects.intmat import Intmat
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.processing_objects.dm import DM
from specula.connections import InputValue
from specula import np


class RecCalibrator(BaseProcessingObj):
    """
    Reconstruction matrix calibrator processing object.
    Analyzes an interaction matrix (Intmat) to compute a reconstruction matrix (Rec).
    """
    
    def __init__(self,
                 nmodes: int,
                 data_dir: str,     # Set by main simul object
                 rec_tag: str,
                 first_mode: int = 0,
                 pupdata_tag: str = None,
                 overwrite: bool = False,
                 mmse: bool = False,
                 r0: float = 0.15,
                 L0: float = 25.0,
                 dm: DM = None,
                 noise_cov: float | np.ndarray | list = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.nmodes = nmodes
        self.first_mode = first_mode
        self.data_dir = data_dir
        self.pupdata_tag = pupdata_tag
        self.overwrite = overwrite

        # Set the MMSE parameters
        self.mmse = mmse
        self.r0 = r0
        self.L0 = L0
        self.dm = dm
        if self.mmse:
            if dm is None:
                raise ValueError('dm must be provided for MMSE reconstruction')
            if noise_cov is None:
                raise ValueError('noise_cov must be provided for MMSE reconstruction')
        if noise_cov is None:
            self.noise_cov = None
        elif isinstance(noise_cov, list):
            self.noise_cov = [self.to_xp(noise_cov_i) for noise_cov_i in noise_cov]
        else:
            self.noise_cov = self.to_xp(noise_cov)

        rec_path = os.path.join(self.data_dir, rec_tag)
        if not rec_path.endswith('.fits'):
            rec_path += '.fits'
        if os.path.exists(rec_path) and not self.overwrite:
            raise FileExistsError(f'REC file {rec_path} already exists, please remove it')
        self.rec_path = rec_path

        self.inputs['in_intmat'] = InputValue(type=Intmat)

    @classmethod
    def input_names(cls):
        return {'in_intmat': InputDesc(Intmat, 'Input interaction matrix to invert')}

    @classmethod
    def output_names(cls):
        return {}

    def finalize(self):
        im = self.local_inputs['in_intmat']

        os.makedirs(self.data_dir, exist_ok=True)

        # TODO add to RM the information about the first mode
        if self.mmse:
            diameter = self.dm.pixel_pitch * self.dm.pixel_pupil
            # The turbulence prior is computed on the modes of the DM input command,
            # starting from first_mode as the interaction matrix columns
            if self.dm.m2c_selected is not None:
                modal_base = IFunc(ifunc=self.dm.ifunc_applied, mask=self.dm.mask,
                                   target_device_idx=self.target_device_idx, precision=self.precision)
                m2c = M2C(self.dm.m2c_selected[:, self.first_mode:],
                        target_device_idx=self.target_device_idx, precision=self.precision)
            else:
                modal_base = IFunc(ifunc=self.dm.ifunc_applied[self.first_mode:], mask=self.dm.mask,
                                   target_device_idx=self.target_device_idx, precision=self.precision)
                m2c = None
            rec = im.generate_rec_mmse(self.r0, self.L0, diameter, modal_base,
                                       self.noise_cov, nmodes=self.nmodes, m2c=m2c)
        else:
            rec = im.generate_rec(self.nmodes)
        rec.save(self.rec_path, overwrite=self.overwrite)
