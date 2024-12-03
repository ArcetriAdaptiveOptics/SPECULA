from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes

    
class Modalrec(BaseProcessingObj):
    '''Modal reconstructor'''

    def __init__(self,
                 nmodes: int=None,
                 recmat: Recmat=None,
                 projmat: Recmat=None,
                 intmat: Intmat=None,
                 polc: bool=False,
                 filtmat = None,
                 identity: bool=False,
                 ncutmodes: int=None,
                 nSlopesToBeDiscarded: int=None,
                 dmNumber: int=0,
                 noProj: bool=False,
                 target_device_idx=None, 
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if polc:
            if identity:
                raise ValueError('identity cannot be set with POLC.')
            if ncutmodes is not None:
                raise ValueError('ncutmodes cannot be set with POLC.')
        else:
            if recmat is None:
                if identity:
                    if nmodes is None:
                        raise ValueError('modalrec nmodes key must be set!')
                    recmat = Recmat(self.xp.identity(nmodes),
                                    target_device_idx=target_device_idx, precision=precision)
                elif intmat:
                    if nmodes:
                        nmodes_intmat = intmat.size[0]
                        intmat.reduce_size(nmodes_intmat - nmodes)
                    if nSlopesToBeDiscarded:
                        intmat.reduce_slopes(nSlopesToBeDiscarded)
                    recmat = Recmat(intmat.intmat,
                                    target_device_idx=target_device_idx, precision=precision)

            if ncutmodes:
                if recmat is not None:
                    recmat.reduce_size(ncutmodes)
                else:
                    print('recmat cannot be reduced because it is null.')


        if recmat is not None:
            if projmat is None and recmat.proj_list and not noProj:
                if dmNumber is not None:
                    if dmNumber <= 0:
                        raise ValueError('dmNumber must be > 0')
                    projmat = Recmat(recmat.proj_list[dmNumber - 1])
                else:
                    raise ValueError('dmNumber (>0) must be defined if projmat_tag is not defined!')

        if filtmat is not None and recmat is not None:
            recmat.recmat = recmat.recmat @ filtmat
            print('recmat updated with filmat!')

        self._recmat = recmat
        self._projmat = projmat
        self._intmat = intmat
        self._polc = polc

        self._layer_modes_list = None
        self._control_list = []
        self._past_step_list = []

        self._modes = BaseValue('output modes from modal reconstructor', target_device_idx=target_device_idx)
        self._pseudo_ol_modes = BaseValue('output POL modes from modal reconstructor', target_device_idx=target_device_idx)
        self._modes_first_step = BaseValue('output (no projection) modes from modal reconstructor', target_device_idx=target_device_idx)

        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.inputs['in_slopes_list'] = InputList(type=Slopes)
        self.outputs['out_modes'] = self.out_modes
        self.outputs['out_pseudo_ol_modes'] = self.pseudo_ol_modes
        self.outputs['out_modes_first_step'] = self.modes_first_step

    @property
    def recmat(self):
        return self._recmat

    @recmat.setter
    def recmat(self, value):
        self._recmat = value

# TODO
    # @in_slopes.setter
    # def in_slopes(self, value: Slopes):
    #     self._slopes = value
    #     if isinstance(value, Slopes):
    #         size = value.size
    #     if self._polc:
    #         self._pseudo_ol_slopes = Slopes(size)

    @property
    def out_modes(self) -> BaseValue:
        return self._modes

    @property
    def projmat(self):
        return self._projmat

    @projmat.setter
    def projmat(self, value):
        self._projmat = value

    @property
    def out_layer_modes_list(self):
        return self._layer_modes_list

    @property
    def modes_first_step(self) -> BaseValue:
        return self._modes_first_step

    @modes_first_step.setter
    def modes_first_step(self, value):
        self._modes_first_step = value

    @property
    def pseudo_ol_slopes(self):
        return self._pseudo_ol_slopes

    @property
    def pseudo_ol_modes(self) -> BaseValue:
        return self._pseudo_ol_modes

    @property
    def intmat(self):
        return self._intmat

    @intmat.setter
    def intmat(self, value):
        self._intmat = value

    @property
    def dm_idx(self):
        return self._dm_idx

    @dm_idx.setter
    def dm_idx(self, value):
        self._dm_idx = value

    def add_control(self, control):
        if not self._polc:
            raise Exception("Control can be added only if POLC is set to 1.")
        self._control_list.append(control)

    def trigger_code(self): # , slope_ptr=None
        if self._recmat.recmat is None:
            print("WARNING: modalrec skipping reconstruction because recmat is NULL")
            return

        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']
        if slopes is None:
            slopes = self.xp.hstack([x.slopes for x in slopes_list])
        else:
            slopes = slopes.slopes
        
        comm_new = []
        if len(self._control_list) > 0:
            for control in self._control_list:
                comm_new.append(control.get_past_state(0))

        if self._modes_first_step.generation_time != self.current_time:
            if self._polc:
                self.compute_pseudo_ol_slopes(self.current_time)
                m = self.compute_modes(self._recmat, self._pseudo_ol_slopes.slopes)
                self._pseudo_ol_modes.value = m
                self._pseudo_ol_modes.generation_time = self.current_time

                if len(m) == len(comm_new):
                    self._modes_first_step.value = m - comm_new
                else:
                    self._modes_first_step.value = m
            else:
                m = self.compute_modes(self._recmat, slopes)
                self._modes_first_step.value = m

            self._modes_first_step.generation_time = self.current_time

            if self._layer_modes_list is not None:
                for i, idx_list in enumerate(self._layer_idx_list):
                    self._layer_modes_list[i].value = self._modes_first_step.value[idx_list]
                    self._layer_modes_list[i].generation_time = self.current_time

        if self._projmat is None:
            if self._verbose:
                n = len(self._modes_first_step.value)
                print(f"(no projmat) first {min(6, n)} residual values: {self._modes_first_step.value[:min(5, n)]}")
            self._modes.value = self._modes_first_step.value
            self._modes.generation_time = self._modes_first_step.generation_time
        else:
            mp = self.compute_modes(self._projmat, self._modes_first_step.ptr_value)
            if self._verbose:
                print(f"first {min(6, len(mp))} residual values after projection: {mp[:min(5, len(mp))]}")
            self._modes.value = mp
            self._modes.generation_time = self.current_time

        if self._polc and len(self._modes_first_step.value) != len(comm_new):
            self._modes.value -= self._control_list[self._dm_idx].get_past_state(0)

    def compute_pseudo_ol_slopes(self, t, slopes=None):
        if slopes is None:
            slopes = self.local_inputs['in_slopes']

        if isinstance(slopes, Slopes):
            self._pseudo_ol_slopes.slopes = slopes.slopes

        comm = []
        for control in self._control_list:
            comm.append(control.get_past_state(0))

        if not self._intmat:
            raise Exception("POLC requires intmat, but it is not set")

        comm_slopes = self.compute_modes(self._intmat, self.xp.array(comm, dtype=self.dtype), intmat=True)
        self._pseudo_ol_slopes.slopes += comm_slopes
        self._pseudo_ol_slopes.generation_time = t

    def compute_modes(self, matrix, slopes_vector=None, intmat=False):

        if slopes_vector is None:
            if isinstance(self._slopes, Slopes):
                slopes_vector = self._slopes.slopes
                if self._verbose:
                    print('Slopes')
                    print(f"modalrec.compute_modes slope RMS: {self.xp.sqrt(self.xp.mean(slope_ptr**2))}")
            elif isinstance(self._slopes, BaseValue):
                slopes_vector = self._slopes.ptr_value
                if self._verbose:
                    print(f"modalrec.compute_modes base_value RMS: {self.xp.sqrt(self.xp.mean(slope_ptr**2))}")
            else:
                raise TypeError('Unsupported self._slopes type:', type(self._slopes))

        if intmat:
            m = slopes_vector @ intmat
        else:
            if len(slopes_vector) != matrix.recmat.shape[1]:
                raise ValueError(f'Cannot multiply slopes vector of length {len(slopes_vector)} with rec. matrix of shape {matrix.recmat.T.shape}')
            m = slopes_vector @ matrix.recmat.T

        return m

    def run_check(self, time_step):
        errmsg = []
        slopes = self.inputs['in_slopes'].get(self.target_device_idx)
        slopes_list = self.inputs['in_slopes_list'].get(self.target_device_idx)
        
        if not slopes and not all(slopes_list):
            errmsg.append("Slopes object not valid")
        if not self._recmat:
            errmsg.append("Recmat object not valid")
        out = bool(slopes or all(slopes_list)) and bool(self._recmat)
        if self._polc:
            if not self._intmat:
                errmsg.append("Intmat object not valid")
            if not self._control_list:
                errmsg.append("ControlList object not valid")
            out &= bool(self._intmat) and bool(self._control_list)
        if len(errmsg) > 0:
            print(", ".join(errmsg))

#        super().build_stream()

        return out



