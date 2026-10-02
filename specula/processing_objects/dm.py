from typing import List

import numpy as np

from specula.base_value import BaseValue
from specula.connections import InputValue

from specula.data_objects.m2c import M2C
from specula.data_objects.ifunc import IFunc
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.data_objects.simul_params import SimulParams

class DM(BaseProcessingObj):
    """
    Deformable Mirror processing object.
    It receives a command vector as input and produces a Layer object representing the DM wavefront.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 height: float,
                 ifunc: IFunc=None,
                 m2c: M2C=None,
                 type_str: str=None,
                 nmodes: int=None,
                 nzern: int=None,
                 start_mode: int=None,
                 idx_modes = None,
                 npixels: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 pupilstop: Pupilstop=None,
                 sign: int=-1,
                 stroke=None,
                 stiffness: np.ndarray=None,
                 max_force: float | List[float]=None,
                 target_device_idx: int=None,
                 precision: int=None
                 ):
        """
        Note
        ----
        - The output layer object contains a wavefront not a surface. Wavefront = 2 x surface (reflection).
        - The DM wavefront deformation is represented as a phase screen in nanometers.
        - Sign parameter is -1 by default to account for reflection in wave propagation.

        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object containing pupil size, pixel pitch, etc.
        height : float [m]
            Height of the DM layer in meters (this is distance from the pupil).
        ifunc : IFunc
            Influence function object defining the DM actuator influence functions.
        m2c : M2C
            Mode-to-command matrix object for converting mode commands to actuator commands.
        type_str : str
            Type of influence function to use if `ifunc` is not provided.
        nmodes : int [1], optional
            Number of modes to consider if `ifunc` is not provided.
        nzern : int [1], optional
            Maximum Zernike radial order if `ifunc` is not provided.
            This is used from mixed Zernike KL bases (not implemented yet).
        start_mode : int [1], optional
            Starting mode index for the DM modes.
        idx_modes : list or array [1], optional
            Specific mode indices to use for the DM. If provided, `start_mode` and `nmodes` are ignored.
        npixels : int [pixels], optional
            Number of pixels for the DM layer. If None, defaults to pupil size.
        obsratio : float [1], optional
            Obscuration ratio for the influence function if `ifunc` is not provided.
        diaratio : float [1], optional
            Diagonal ratio for the influence function if `ifunc` is not provided.
        pupilstop : Pupilstop
            Pupilstop object defining the DM aperture.
        sign : int [1], optional
            Sign for the DM surface deformation, by default -1.
        stroke : float or list [nm], optional 
            The maximum amplitude (in NANOMETERS) to which commands are clipped at. 
            If a list is given, this is the maximum amplitude that can be applied per mode.
            Default is None (no clipping applied).
        stiffness : array [force/command unit], optional
            Stiffness matrix (nact x nact), with nact equal to the number of m2c rows: forces are
            computed as ``stiffness @ command``, with the command after m2c. Units are up to
            the user (e.g. N per command unit); in a YAML file use ``stiffness_data``.
            Requires m2c. If given, the forces of the applied command are output
            as ``out_forces``. Default is None (no force computation).
        max_force : float or list [force], optional
            Maximum absolute force per actuator (scalar or one value per actuator),
            in the same force unit as stiffness.
            Requires stiffness. If the forces exceed it, the highest-order modes are
            discarded: the largest number of modes (starting from the first one) whose
            forces are within the limit is kept, down to a zero command.
            This assumes that the modes are sorted by increasing spatial frequency
            (e.g. KL modes). Force limiting is applied before the stroke clipping:
            if both are active, the stroke clipping changes the actuator positions and
            the resulting forces may slightly exceed max_force.
            Default is None (no force limiting).
        target_device_idx : int [1], optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int [1], optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
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
            nmodes = self._ifunc.nmodes()

        if idx_modes is not None:
            self._valid_modes = idx_modes
            self.n_valid_modes = len(idx_modes)
        else:
            self._valid_modes = slice(start_mode, nmodes)
            self.n_valid_modes = len(range(start_mode, nmodes))

        if m2c is not None:
            if m2c.m2c.shape[0] != self._ifunc.nmodes():
                raise ValueError(f'm2c has {m2c.m2c.shape[0]} rows, but the influence function '
                                 f'has {self._ifunc.nmodes()} modes')
            self.m2c = m2c.m2c
            nmodes_m2c = m2c.m2c[:, self._valid_modes].shape[1]
            self.m2c_commands = self.xp.zeros(nmodes_m2c, dtype=self.dtype)
            out_comm_len = self.m2c.shape[0]
        else:
            self.m2c = None
            self.m2c_commands = None
            out_comm_len = self.n_valid_modes
        
        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.nmodes()
        self.if_commands = self.xp.zeros(nmodes_if, dtype=self.dtype)

        self.if_commands_selector = slice(0, self.n_valid_modes)

        self.layer = Layer(s[0], s[1], self.pixel_pitch, height, target_device_idx=target_device_idx, precision=precision)
        self.layer.A = self._ifunc.mask_inf_func

        self.nmodes = nmodes - start_mode   # Input command vector is not supposed to include the modes before "start_mode"

        # Default sign is -1 to take into account the reflection in the propagation
        self.sign = sign

        self.stroke = None
        if stroke is not None:
            if isinstance(stroke,list):
                if out_comm_len != len(stroke):
                    raise ValueError(f'Stroke is a list of {len(stroke)} elements, but {out_comm_len} coefficients are expected')
                self.stroke = self.xp.array(stroke, dtype=self.dtype)
            else:
                self.stroke = self.xp.ones(out_comm_len, dtype=self.dtype) * self.xp.asarray(stroke, dtype=self.dtype)
        
        self.stiffness = None
        self.max_force = None
        if stiffness is not None:
            if self.m2c is None:
                raise ValueError('stiffness requires m2c')
            self.stiffness = self.to_xp(stiffness, dtype=self.dtype)
            if self.stiffness.shape != (out_comm_len, out_comm_len):
                raise ValueError(f'stiffness has shape {self.stiffness.shape}, '
                                 f'but ({out_comm_len}, {out_comm_len}) is expected')
        if max_force is not None:
            if self.stiffness is None:
                raise ValueError('max_force requires stiffness')
            if isinstance(max_force, list) and len(max_force) != out_comm_len:
                raise ValueError(f'max_force is a list of {len(max_force)} elements, '
                                 f'but {out_comm_len} actuators are expected')
            self.max_force = self.xp.ones(out_comm_len, dtype=self.dtype) * \
                             self.xp.asarray(max_force, dtype=self.dtype)
            # Force of each mode for a unit coefficient, one column per mode
            self._mode_forces = self.stiffness @ self.m2c[:, self._valid_modes]
            self._mode_idx = self.xp.arange(len(self.m2c_commands))

        self.forces = BaseValue(
            target_device_idx=target_device_idx,
            precision=precision,
            value=self.xp.zeros(out_comm_len if self.stiffness is not None else 0, dtype=self.dtype)
        )
        # All modes are kept unless the force limiting discards some
        nmodes_cmd = len(self.m2c_commands) if self.m2c is not None else self.n_valid_modes
        self.force_nmodes = BaseValue(
            target_device_idx=target_device_idx,
            value=self.xp.full(1, nmodes_cmd, dtype=self.xp.int64)
        )

        self.clip_command = BaseValue(
            target_device_idx=target_device_idx,
            precision=precision,
            value=self.xp.zeros(out_comm_len, dtype=self.dtype)
        )
        self.outputs['out_clipped_command'] = self.clip_command
        self.outputs['out_forces'] = self.forces
        self.outputs['out_force_nmodes'] = self.force_nmodes
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.outputs['out_layer'] = self.layer

    @classmethod
    def input_names(cls):
        return {'in_command': InputDesc(BaseValue, 'Input command vector for DM actuators')}

    @classmethod
    def output_names(cls):
        return {'out_layer': OutputDesc(Layer, 'Output wavefront layer produced by the DM'),
                'out_clipped_command': OutputDesc(BaseValue, 'DM applied command, after (optional) clipping'),
                'out_forces': OutputDesc(BaseValue, 'Actuator forces of the applied command (empty without stiffness)'),
                'out_force_nmodes': OutputDesc(BaseValue, 'Number of modes kept by the force limiting')}

    def trigger_code(self):
        input_commands = self.local_inputs['in_command'].value[:self.nmodes]

        if self.m2c is not None:
            self.m2c_commands[:len(input_commands)] = input_commands
            if self.max_force is not None:
                limited_commands, n_keep = self._limit_forces(self.m2c_commands)
                self.m2c_commands[:] = limited_commands
                self.force_nmodes.value[0] = n_keep
                self.force_nmodes.generation_time = self.current_time
            cmd = self.m2c[:, self._valid_modes] @ self.m2c_commands
        else:
            cmd = input_commands
        # Perform clipping
        if self.stroke is not None: 
            cmd = self.xp.minimum(self.xp.maximum(-self.stroke[:len(cmd)],cmd),self.stroke[:len(cmd)])
        self.if_commands[:len(cmd)] = self.sign * cmd

        if self.m2c is not None:
            self.layer.phaseInNm[self._ifunc.idx_inf_func] = self.if_commands @ self._ifunc.influence_function
        else:
            self.layer.phaseInNm[self._ifunc.idx_inf_func] = \
                self.if_commands[self.if_commands_selector] @ self._ifunc.influence_function[self._valid_modes, :]
        self.layer.generation_time = self.current_time
        self.clip_command.value[:len(cmd)] = cmd
        self.clip_command.generation_time = self.current_time
        if self.stiffness is not None:
            self.forces.value[:] = self.stiffness @ self.clip_command.value
            self.forces.generation_time = self.current_time

    def _limit_forces(self, commands):
        '''
        Return the modal commands with the highest-order modes discarded until the
        forces are within max_force, and the number of modes kept. The cumulative sum
        gives the forces of all the possible truncations at once, without iterations
        or host synchronization.
        '''
        forces = self.xp.cumsum(self._mode_forces * commands, axis=1)
        ok = self.xp.all(self.xp.abs(forces) <= self.max_force[:, None], axis=0)
        # ok[n-1]: forces within limits keeping n modes. Keeping 0 modes is always ok.
        ok = self.xp.concatenate((self.xp.ones(1, dtype=bool), ok))
        n_keep = len(ok) - 1 - self.xp.argmax(ok[::-1])
        return commands * (self._mode_idx < n_keep), n_keep

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
