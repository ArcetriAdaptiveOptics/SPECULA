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
            End index (exclusive) of the selected modes: m2c columns if m2c is given,
            otherwise influence function rows. Default: all of them.
            If `ifunc` is not provided, it is also the number of generated modes.
        nzern : int [1], optional
            Maximum Zernike radial order if `ifunc` is not provided.
            This is used from mixed Zernike KL bases (not implemented yet).
        start_mode : int [1], optional
            Index of the first selected mode. The input command starts from this mode.
        idx_modes : list or array [1], optional
            Specific mode indices to use for the DM. Cannot be set together with `start_mode` or `nmodes`.
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
            If a list is given, this is the maximum amplitude that can be applied per actuator,
            i.e. per applied influence function row: all the m2c rows if m2c is given,
            otherwise the selected modes.
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
            # start_mode and idx_modes are not passed to IFunc because they are handled by self._sel
            ifunc = IFunc(type_str=type_str, mask=mask, npixels=npixels,
                           obsratio=obsratio, diaratio=diaratio, nzern=nzern,
                           nmodes=nmodes,
                           target_device_idx=target_device_idx, precision=precision)
        self._ifunc = ifunc
        self.tag = self._ifunc.tag

        if m2c is not None:
            if m2c.m2c.shape[0] != self._ifunc.nmodes():
                raise ValueError(f'm2c has {m2c.m2c.shape[0]} rows, but the influence function '
                                 f'has {self._ifunc.nmodes()} modes')
            self.m2c = m2c.m2c
        else:
            self.m2c = None

        # Mode selection: m2c columns with m2c, influence function rows without m2c
        n_basis = self.m2c.shape[1] if self.m2c is not None else self._ifunc.nmodes()
        if idx_modes is not None:
            self._sel = idx_modes
        else:
            if nmodes is None:
                nmodes = n_basis
            if nmodes > n_basis:
                raise ValueError(f'nmodes={nmodes} exceeds the {n_basis} available modes')
            self._sel = slice(start_mode or 0, nmodes)
        self._apply_mode_selection()

        # Input command length (the input does not include the modes before start_mode)
        self.nmodes = self._m2c_sel.shape[1] if self.m2c is not None else self._ifunc_act.shape[0]
        if self.nmodes == 0:
            raise ValueError(f'The mode selection is empty (start_mode={start_mode}, nmodes={nmodes}, '
                             f'idx_modes={idx_modes})')
        # Number of actuators, i.e. of applied influence function rows
        n_act = self._ifunc_act.shape[0]
        self._modes = self.xp.zeros(self.nmodes, dtype=self.dtype)

        s = self._ifunc.mask_inf_func.shape
        self.layer = Layer(s[0], s[1], self.pixel_pitch, height, target_device_idx=target_device_idx, precision=precision)
        self.layer.A = self._ifunc.mask_inf_func

        # Default sign is -1 to take into account the reflection in the propagation
        self.sign = sign

        self.stroke = None
        if stroke is not None:
            if isinstance(stroke,list):
                if n_act != len(stroke):
                    raise ValueError(f'Stroke is a list of {len(stroke)} elements, but {n_act} coefficients are expected')
                self.stroke = self.xp.array(stroke, dtype=self.dtype)
            else:
                self.stroke = self.xp.ones(n_act, dtype=self.dtype) * self.xp.asarray(stroke, dtype=self.dtype)

        self.stiffness = None
        self.max_force = None
        if stiffness is not None:
            if self.m2c is None:
                raise ValueError('stiffness requires m2c')
            self.stiffness = self.to_xp(stiffness, dtype=self.dtype)
            if self.stiffness.shape != (n_act, n_act):
                raise ValueError(f'stiffness has shape {self.stiffness.shape}, '
                                 f'but ({n_act}, {n_act}) is expected')
        if max_force is not None:
            if self.stiffness is None:
                raise ValueError('max_force requires stiffness')
            if isinstance(max_force, list) and len(max_force) != n_act:
                raise ValueError(f'max_force is a list of {len(max_force)} elements, '
                                 f'but {n_act} actuators are expected')
            self.max_force = self.xp.ones(n_act, dtype=self.dtype) * \
                             self.xp.asarray(max_force, dtype=self.dtype)
            # Force of each mode for a unit coefficient, one column per mode
            self._mode_forces = self.stiffness @ self._m2c_sel
            self._mode_idx = self.xp.arange(self.nmodes)

        self.forces = BaseValue(
            target_device_idx=target_device_idx,
            precision=precision,
            value=self.xp.zeros(n_act if self.stiffness is not None else 0, dtype=self.dtype)
        )
        # All modes are kept unless the force limiting discards some
        self.force_nmodes = BaseValue(
            target_device_idx=target_device_idx,
            value=self.xp.full(1, self.nmodes, dtype=self.xp.int64)
        )

        self.clip_command = BaseValue(
            target_device_idx=target_device_idx,
            precision=precision,
            value=self.xp.zeros(n_act, dtype=self.dtype)
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
        # A short input is zero-filled, the modes beyond self.nmodes are ignored
        input_commands = self.local_inputs['in_command'].value[:self.nmodes]
        self._modes[:] = 0
        self._modes[:len(input_commands)] = input_commands

        if self.m2c is not None:
            if self.max_force is not None:
                limited_commands, n_keep = self._limit_forces(self._modes)
                self._modes[:] = limited_commands
                self.force_nmodes.value[0] = n_keep
                self.force_nmodes.generation_time = self.current_time
            cmd = self._m2c_sel @ self._modes
        else:
            cmd = self._modes
        # Perform clipping
        if self.stroke is not None:
            cmd = self.xp.minimum(self.xp.maximum(-self.stroke, cmd), self.stroke)

        self.layer.phaseInNm[self._ifunc.idx_inf_func] = (self.sign * cmd) @ self._ifunc_act
        self.layer.generation_time = self.current_time
        self.clip_command.value[:] = cmd
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

    def _apply_mode_selection(self):
        '''
        Apply the mode selection once, so that the trigger does not copy the
        selected m2c columns or influence function rows at every step.
        With m2c the selection is folded into the m2c columns and all the
        influence function rows are applied; without m2c only the selected rows are.
        _ifunc_act is a view or a copy of the influence function: it is kept in sync
        by the ifunc setter, not by direct changes to ifunc_obj.influence_function.
        '''
        if self.m2c is not None:
            self._m2c_sel = self.xp.ascontiguousarray(self.to_xp(self.m2c[:, self._sel]))
            self._ifunc_act = self.to_xp(self._ifunc.influence_function)
        else:
            self._m2c_sel = None
            self._ifunc_act = self.xp.ascontiguousarray(
                self.to_xp(self._ifunc.influence_function[self._sel]))

    # Getters and Setters for the attributes
    @property
    def ifunc(self):
        return self._ifunc.influence_function

    @property
    def ifunc_obj(self):
        """Return the IFunc object (not just the array)"""
        return self._ifunc

    @property
    def ifunc_applied(self):
        """Influence function rows that are applied: all of them with m2c,
        the selected modes without m2c"""
        return self._ifunc_act

    @property
    def m2c_selected(self):
        """m2c columns of the selected modes (None without m2c): column j is
        driven by the input command element j"""
        return self._m2c_sel

    @ifunc.setter
    def ifunc(self, value):
        # The selection, stroke, stiffness and outputs are sized on the current shape
        if value.shape != self._ifunc.influence_function.shape:
            raise ValueError(f'ifunc has shape {value.shape}, but '
                             f'{self._ifunc.influence_function.shape} is expected')
        self._ifunc.influence_function = value
        self._apply_mode_selection()

    @property
    def mask(self):
        return self._ifunc.mask_inf_func

    @property
    def type_str(self):
        return self._ifunc.type_str
