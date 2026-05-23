from collections import namedtuple

from specula import cpuArray, process_comm
from specula import np, cp
from specula.lib.utils import flatten
from specula.log import get_specula_logger


Output = namedtuple('Output', 'obj_name output_key delay ref input_name type')


def split_output(output_name, use_inputs=False, detect_types=False):
    '''
    Split the output name into object name and output key.
    '''
    if ':' in output_name:
        output_name, suffix = output_name.split(':')
        if detect_types:
            if suffix in ['float', 'int', 'str']:
                delay = 0
                typ_ = suffix
            else:
                raise ValueError(f'Unknown type {suffix}')
        else:
            delay = int(suffix)
            typ_ = None
    else:
        delay = 0
        typ_ = None

    if '-' in output_name:
        input_name, output_name = output_name.split('-')
    else:
        input_name = None

    if '.' in output_name:
        obj_name, output_key = output_name.split('.')
    else:
        obj_name = output_name
        output_key = None

    return Output(obj_name, output_key, delay, None, input_name, typ_)


class _InputItem():
    def __init__(self, type_, value, remote_rank=None, tag=None, optional=False,
                 requesting_obj_name=None, input_name=None):
        """
        Private class, wrapper for simple input values

        *value* must be a reference to the output value being read, or None
        in case of remote inputs.
        """
        self.logger = get_specula_logger(__name__)
        if remote_rank is not None:
            if value is not None:
                raise ValueError(f'non-None value used with remote input')
        else:
            if not isinstance(value, type_):
                raise ValueError(f'Value must be of type {type_} instead of {type(value)}')

        self.type = type_
        self.cloned_value = None
        self.optional = optional
        self.remote_rank = remote_rank
        self.tag = tag
        self.output_ref = value
        self.requesting_obj_name = requesting_obj_name
        self.input_name = input_name

    def receive_new_value(self, first_mpi_receive=True):
        self.logger.mpi_send_debug(
                               f'RECV from rank {self.remote_rank} {self.tag=} type={self.type})'
                               )
        if first_mpi_receive or self.cloned_value.get_value() is None:
            self.logger.mpi_send_debug(f'recv with Pickle tag={self.tag}')
            new_value = process_comm.recv(source=self.remote_rank, tag=self.tag)
            if new_value.xp_str == 'cp':
                new_value.xp = cp
            else:
                new_value.xp = np
        else:            
            self.logger.mpi_send_debug(f'Recv with Buffer')
            new_value = self.cloned_value
            buffer = cpuArray(self.cloned_value.get_value())
            self.logger.mpi_send_debug(f'tag={self.tag} RECV .buffer ' + str(type(buffer)))
            self.logger.mpi_send_debug(f'tag={self.tag} RECV .buffer dtype' + str(buffer.dtype))
            process_comm.Recv(buffer, source=self.remote_rank, tag=self.tag)
            self.logger.mpi_send_debug(f'tag={self.tag+1} RECV .bufftimeer')
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
            value = self.receive_new_value(first_mpi_receive=self.cloned_value is None)

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
        self.type = type
        self.input_values = []
        self.optional = optional
        self.requesting_obj_name = None
        self.input_name = None

    def get(self, target_device_idx):
        return list(flatten(v.get(target_device_idx) for v in self.input_values))

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

        if not isinstance(item, self.type) and remote_rank is None:
            raise ValueError(f'Item must be of type {self.type} instead of {type(item)}')

        self.input_values.append(_InputItem(self.type,
                                            item,
                                            remote_rank=remote_rank,
                                            tag=tag,
                                            optional=self.optional,
                                            requesting_obj_name=self.requesting_obj_name,
                                            input_name=self.input_name))


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
                obj_info = f" (from {self.requesting_obj_name}.{self.input_name})" \
                           if self.requesting_obj_name else ""
                raise ValueError(f'Input {self.input_name} of object {self.requesting_obj_name} is empty and not optional. '
                                 f'Input type: {self.type}{obj_info}')
        return values_list[0]
