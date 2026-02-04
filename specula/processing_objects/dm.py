
from specula.base_value import BaseValue
from specula.connections import InputValue

from specula.data_objects.m2c import M2C
from specula.data_objects.ifunc import IFunc
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams

class DM(BaseProcessingObj):
    """Deformable Mirror processing object
    It receives a command vector as input and produces a Layer object representing the DM wavefront.
    
    Notes:
    - The output layer object contains a wavefront not a surface. Wavefront = 2 x surface (reflection).
    - The DM wavefront deformation is represented as a phase screen in nanometers.
    - Sign parameter is -1 by default to account for reflection in wave propagation.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object containing pupil size, pixel pitch, etc.
    height : float
        Height of the DM layer in meters (this is distance from the pupil).
    ifunc : IFunc, optional
        Influence function object defining the DM actuator influence functions.
    m2c : M2C, optional
        Mode-to-command matrix object for converting mode commands to actuator commands.
    type_str : str, optional
        Type of influence function to use if `ifunc` is not provided.
    nmodes : int, optional
        Number of modes to consider if `ifunc` is not provided.
    nzern : int, optional
        Maximum Zernike radial order if `ifunc` is not provided.
        This is used from mixed Zernike KL bases (not implemented yet).
    start_mode : int, optional
        Starting mode index for the DM modes.
    input_offset : int, optional
        Offset in the input command vector to start reading from, by default 0.
    idx_modes : list or array, optional
        Specific mode indices to use for the DM. If provided, `start_mode` and `nmodes` are ignored.
    npixels : int, optional
        Number of pixels for the DM layer. If None, defaults to pupil size.
    obsratio : float, optional
        Obscuration ratio for the influence function if `ifunc` is not provided.
    diaratio : float, optional
        Diagonal ratio for the influence function if `ifunc` is not provided.
    pupilstop : Pupilstop, optional
        Pupilstop object defining the DM aperture.
    sign : int, optional
        Sign for the DM surface deformation, by default -1 (to account for reflection).
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None (uses global setting).
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None (uses global setting).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 height: float,          # TODO =0.0,
                 ifunc: IFunc=None,
                 m2c: M2C=None,
                 type_str: str=None,
                 nmodes: int=None,
                 nzern: int=None,
                 start_mode: int=None,
                 input_offset: int=0,
                 idx_modes = None,
                 npixels: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 pupilstop: Pupilstop=None,
                 sign: int=-1,
                 target_device_idx: int=None,
                 precision: int=None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.pixel_pupil = self.simul_params.pixel_pupil

        mask = None
        if pupilstop:
            mask = pupilstop.A
            if npixels is not None and mask.shape != (npixels, npixels):
                raise ValueError(f'npixels={npixels} is not consistent with the pupilstop shape {mask.shape}')
            else:
                npixels = mask.shape[0]

        if npixels is not None and ifunc is not None:
            if ifunc.mask_inf_func.shape != (npixels, npixels):
                raise ValueError(f'npixels={npixels} is not consistent with the ifunc shape {ifunc.mask_inf_func.shape}')

        if mask is None and ifunc is None and npixels is None:
            npixels = self.pixel_pupil

        if idx_modes is not None:
            if start_mode is not None:
                raise ValueError('start_mode cannot be set together with idx_modes.')
            if nmodes is not None:
                raise ValueError('nmodes cannot be set together with idx_modes.')

        if not ifunc:
            if nmodes is None and idx_modes is not None:
                nmodes = max(idx_modes) + 1
            # start_mode and idx_modes are not passed to IFunc because they are handled by self._valid_modes
            ifunc = IFunc(type_str=type_str, mask=mask, npixels=npixels,
                           obsratio=obsratio, diaratio=diaratio, nzern=nzern,
                           nmodes=nmodes,
                           target_device_idx=target_device_idx, precision=precision)
        self._ifunc = ifunc
        self.tag = self._ifunc.tag

        if start_mode is None:
            start_mode = 0
        if nmodes is None:
            nmodes = self._ifunc.size[0]

        if idx_modes is not None:
            self._valid_modes = idx_modes
            self.n_valid_modes = len(idx_modes)
        else:
            self._valid_modes = slice(start_mode, nmodes)
            self.n_valid_modes = len(range(start_mode, nmodes))

        if m2c is not None:
            self.m2c = m2c.m2c
            nmodes_m2c = m2c.m2c[:, self._valid_modes].shape[1]
            self.m2c_commands = self.xp.zeros(nmodes_m2c, dtype=self.dtype)
        else:
            self.m2c = None
            self.m2c_commands = None
        
        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.size[0]
        self.if_commands = self.xp.zeros(nmodes_if, dtype=self.dtype)

        self.if_commands_selector = slice(0, self.n_valid_modes)

        self.layer = Layer(s[0], s[1], self.pixel_pitch, height, target_device_idx=target_device_idx, precision=precision)
        self.layer.A = self._ifunc.mask_inf_func

        self.input_offset = input_offset
        self.nmodes = nmodes - start_mode   # Input command vector is not supposed to include the modes before "start_mode"

        # Default sign is -1 to take into account the reflection in the propagation
        self.sign = sign
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.outputs['out_layer'] = self.layer

    def trigger_code(self):
        input_commands = self.local_inputs['in_command'].value

        if self.nmodes is not None:
            input_commands = input_commands[self.input_offset: self.input_offset + self.nmodes]

        if self.m2c is not None:
            self.m2c_commands[:len(input_commands)] = input_commands
            cmd = self.m2c[:, self._valid_modes] @ self.m2c_commands
        else:
            cmd = input_commands
        self.if_commands[:len(cmd)] = self.sign * cmd

        if self.m2c is not None:
            self.layer.phaseInNm[self._ifunc.idx_inf_func] = self.if_commands @ self._ifunc.influence_function
        else:
            self.layer.phaseInNm[self._ifunc.idx_inf_func] = \
                self.if_commands[self.if_commands_selector] @ self._ifunc.influence_function[self._valid_modes, :]
        self.layer.generation_time = self.current_time

    # Getters and Setters for the attributes
    @property
    def ifunc(self):
        return self._ifunc.influence_function

    @property
    def ifunc_obj(self):
        """Return the IFunc object (not just the array)"""
        return self._ifunc

    @ifunc.setter
    def ifunc(self, value):
        self._ifunc.influence_function = value

    @property
    def mask(self):
        return self._ifunc.mask_inf_func

    @property
    def type_str(self):
        return self._ifunc.type_str
