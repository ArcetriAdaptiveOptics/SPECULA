import os
from typing import Union

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intmat import Intmat
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.processing_objects.dm import DM
from specula.connections import InputValue
from specula import np


class RecCalibrator(BaseProcessingObj):
    def __init__(self,
                 nmodes: int,         # TODO =0,
                 data_dir: str,       # TODO = "",         # Set by main simul object
                 rec_tag: str,        # TODO = "",
                 first_mode: int = 0,
                 pupdata_tag: str = None,
                 tag_template: str = None,
                 overwrite: bool = False,
                 mmse: bool = False,
                 r0: float = 0.15,
                 L0: float = 25.0,
                 dm: DM = None,
                 noise_cov: Union[float, np.ndarray, list] = None,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.nmodes = nmodes
        self.first_mode = first_mode
        self.data_dir = data_dir
        if tag_template is None and (rec_tag is None or rec_tag == 'auto'):
            raise ValueError('At least one of tag_template and rec_tag must be set')
        self.pupdata_tag = pupdata_tag
        self.overwrite = overwrite

        # Set the MMSE parameters
        self.mmse = mmse
        self.r0 = r0
        self.L0 = L0
        self.dm = dm
        if noise_cov is None:
            if self.mmse:
                raise ValueError('noise_cov must be provided for MMSE reconstruction')
            self.noise_cov = None
        elif isinstance(noise_cov, list):
            self.noise_cov = [self.to_xp(noise_cov_i) for noise_cov_i in noise_cov]
        else:
            self.noise_cov = self.to_xp(noise_cov)

        if rec_tag is None or rec_tag == 'auto':
            rec_filename = tag_template
        else:
            rec_filename = rec_tag

        rec_path = os.path.join(self.data_dir, rec_filename)
        if not rec_path.endswith('.fits'):
            rec_path += '.fits'
        if os.path.exists(rec_path) and not self.overwrite:
            raise FileExistsError(f'REC file {rec_path} already exists, please remove it')
        self.rec_path = rec_path

        self.inputs['in_intmat'] = InputValue(type=Intmat)

    def finalize(self):
        im = self.local_inputs['in_intmat']

        os.makedirs(self.data_dir, exist_ok=True)

        # TODO add to RM the information about the first mode
        if self.mmse:
            diameter = self.dm.pixel_pitch * self.dm.pixel_pupil
            modal_base = IFunc(ifunc=self.dm.ifunc, mask=self.dm.mask,
                               target_device_idx=self.target_device_idx, precision=self.precision)
            if self.dm.m2c is not None:
                m2c = M2C(self.dm.m2c,
                        target_device_idx=self.target_device_idx, precision=self.precision)
            else:
                m2c = None
            rec = im.generate_rec_mmse(self.r0, self.L0, diameter, modal_base,
                                       self.noise_cov, nmodes=self.nmodes, m2c=m2c)
        else:
            rec = im.generate_rec(self.nmodes)
        rec.save(self.rec_path, overwrite=self.overwrite)
