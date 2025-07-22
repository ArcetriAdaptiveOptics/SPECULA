from specula import process_rank, process_comm, MPI_DBG, MPI_SEND_DBG
from specula import np, cp
from specula.lib.flatten import flatten


class _InputItem():
    def __init__(self, type, remote_rank=None, tag=None, optional=False):
        """
        Private class, wrapper for simple input values
        """
        self.output_ref_type = type
        self.output_ref = None
        self.cloned_value = None
        self.optional = optional
        self.remote_rank = remote_rank
        self.tag = tag
        self.last_value = None

    def get(self, target_device_idx):
        if self.remote_rank is None:
            if self.output_ref is None:
                self.last_value = None
                return None

            elif self.output_ref.target_device_idx == target_device_idx:
                self.last_value = self.output_ref
                return self.output_ref

        if self.remote_rank is None:         
            value = self.output_ref
        else:
            if MPI_SEND_DBG: print(process_rank, f'RECV from rank {self.remote_rank} {self.tag=} type={self.output_ref_type})', flush=True)
            value = process_comm.recv(source=self.remote_rank, tag=self.tag)
            if value.xp_str == 'cp':
                value.xp = cp
            else:
                value.xp = np

        if self.cloned_value is None:
            self.cloned_value = value.copyTo(target_device_idx)
        else:
            value.transferDataTo(self.cloned_value)

        self.last_value = self.cloned_value
        return self.cloned_value

    def set(self, value):
        if self.output_ref is not None:
            raise ValueError('InputValue already set, cannot set again')        
        if not isinstance(value, self.output_ref_type) and self.remote_rank is None:
            raise ValueError(f'Value must be of type {self.output_ref_type} instead of {type(value)}')
        self.output_ref = value


class InputList():
    def __init__(self, type, optional=False):
        """
        Wrapper for input lists
        """
        self.output_ref_type = type
        self.input_values = []
        self.optional = optional

    def get(self, target_device_idx, single_value=False):
        values_list = flatten([v.get(target_device_idx) for v in self.input_values])
        if single_value:
            if len(values_list) > 1:
                raise ValueError('InputValue contains more than one item')
            if len(values_list) == 0:
                if self.optional:
                    return None
                else:
                    raise ValueError('InputValue is empty and not optional')
            return values_list[0]
        else:
            return values_list

    def set(self, item, remote_rank=None, tag=None):
        """
        Set a single item as the input list
        """
        self.input_values = []
        self.append(item, remote_rank, tag)

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
                                            remote_rank=remote_rank,
                                            tag=tag,
                                            optional=self.optional))
        self.input_values[-1].set(item)


class InputValue(InputList):
    '''
    Convenience class for input lists. Calling get() will return a list of items.
    '''
    def get(self, target_device_idx):
        return super().get(target_device_idx, single_value=True)
