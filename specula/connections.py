from specula import cpuArray, process_rank, process_comm, MPI_DBG, MPI_SEND_DBG
from specula import np, cp
from specula.lib.flatten import flatten


class _InputItem():
    def __init__(self, type_, value, remote_rank=None, tag=None, optional=False):
        """
        Private class, wrapper for simple input values

        *value* must be a reference to the output value being read, or None
        in case of remote inputs.
        """
        if remote_rank is not None:
            if value is not None:
                raise ValueError(f'non-None value used with remote input')
        else:
            if not isinstance(value, type_):
                raise ValueError(f'Value must be of type {type_} instead of {type(value)}')

        self.output_ref_type = type_
        self.cloned_value = None
        self.optional = optional
        self.remote_rank = remote_rank
        self.tag = tag
        self.output_ref = value


    def receive_new_value(self, first_mpi_receive=True):
        if MPI_SEND_DBG: print(process_rank, f'RECV from rank {self.remote_rank} {self.tag=} type={self.output_ref_type})', flush=True)
        if first_mpi_receive or self.cloned_value.get_value() is None:
            if MPI_SEND_DBG: print(process_rank, f'recv with Pickle', self.tag, flush=True)
            new_value = process_comm.recv(source=self.remote_rank, tag=self.tag)
            if new_value.xp_str == 'cp':
                new_value.xp = cp
            else:
                new_value.xp = np
        else:            
            if MPI_SEND_DBG: print(process_rank, f'Recv with Buffer', flush=True)
            new_value = self.cloned_value
            buffer = cpuArray(self.cloned_value.get_value())
            if MPI_SEND_DBG:  print(process_rank, self.tag, 'RECV .buffer', type(buffer))
            if MPI_SEND_DBG:  print(process_rank, self.tag, 'RECV .buffer dtype', buffer.dtype)
            process_comm.Recv(buffer, source=self.remote_rank, tag=self.tag)
            if MPI_SEND_DBG:  print(process_rank, self.tag+1, 'RECV .bufftimeer')
            gen_time = process_comm.recv(source=self.remote_rank, tag=self.tag+1)
            self.cloned_value.generation_time = gen_time
            self.cloned_value.set_value(buffer)

        return new_value        

    def get(self, target_device_idx):
        if self.remote_rank is None:
            if self.output_ref is None:
                self.cloned_value = None
                return None

            elif self.output_ref.target_device_idx == target_device_idx:
                self.cloned_value = self.output_ref
                return self.cloned_value

        if self.remote_rank is None:         
            value = self.output_ref
        else:
            value = self.receive_new_value(first_mpi_receive=self.cloned_value is None )

        if self.cloned_value is None:
            self.cloned_value = value.copyTo(target_device_idx)
        else:
            value.transferDataTo(self.cloned_value)
        return self.cloned_value


class InputList():
    def __init__(self, type, optional=False):
        """
        Wrapper for input lists exchanged by objects. All inputs and outputs
        are managed as lists. Singles values use the InputValue() class below,
        which just reduces to a list with a single value.

        Each list element is a separate _InputItem instance, which is able to
        perform its own MPI receive if needed. This allows to mix in the same list
        inputs with different sources (useful e.g. in propagation)
        """
        self.output_ref_type = type
        self.input_values = []
        self.optional = optional

    def get(self, target_device_idx):
        return flatten([v.get(target_device_idx) for v in self.input_values])

    def set(self, values_list, remote_rank=None, tag=None):
        """
        Set the input values for the list.
        """
        self.input_values = []
        self.append(values_list, remote_rank, tag)

    def append(self, item, remote_rank=None, tag=None):
        """
        Append an item to the input list, optionally specifying a remote rank and tag.
        If the item is a list, it will be flattened and each item will be added to the input list.
        """
        if isinstance(item, list):
            for v in item:
                self.append(v, remote_rank, tag)
            return

        if not isinstance(item, self.output_ref_type) and remote_rank is None:
            raise ValueError(f'Item must be of type {self.output_ref_type} instead of {type(item)}')

        self.input_values.append(_InputItem(self.output_ref_type,
                                            item,
                                            remote_rank=remote_rank,
                                            tag=tag,
                                            optional=self.optional))


class InputValue(InputList):
    '''
    Convenience class for single values: calling get() will return a single item
    '''
    def set(self, item, remote_rank=None, tag=None):
        """
        Set a single item as the input list
        """
        self.input_values = []
        self.append(item, remote_rank, tag)

    def get(self, target_device_idx):
        values_list = super().get(target_device_idx)
        if len(values_list) > 1:
            raise ValueError('InputValue contains more than one item')
        if len(values_list) == 0:
            if self.optional:
                return None
            else:
                raise ValueError('InputValue is empty and not optional')
        return values_list[0]

