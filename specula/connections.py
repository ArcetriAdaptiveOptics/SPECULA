from specula import process_rank, process_comm, MPI_DBG
from specula import np, cp
from specula.lib.flatten import flatten

class InputValue():
    def __init__(self, type, optional=False):
        """
        Wrapper for simple input values
        """
        self.output_ref_type = type
        self.output_ref = None
        self.cloned_value = None
        self.optional = optional
        self.remote_rank = None
        self.remote = False
        self.tag = None

    def set_tag(self, v):
        self.tag = v        

    def set_remote_rank(self, remote_rank):
        self.remote = remote_rank is not None
        # the sender rank
        self.remote_rank = remote_rank

    def get_time(self):
        if not self.output_ref is None:
            return self.output_ref.generation_time        

    def get(self, target_device_idx):
        if not self.remote:            
            if not self.output_ref is None:            
                if self.output_ref.target_device_idx == target_device_idx:
                    return self.output_ref
                else:
                    if self.cloned_value is None:
                        if type(self.output_ref) is list:
                            # if the output_ref is a list, we need to copy each element
                            self.cloned_value = [v.copyTo(target_device_idx) for v in self.output_ref]
                        else:
                            self.cloned_value = self.output_ref.copyTo(target_device_idx)
                    else:
                        if type(self.output_ref) is list:
                            # if the output_ref is a list, we need to transfer each element
                            for output, cloned in zip(self.output_ref, self.cloned_value):
                                output.transferDataTo(cloned)
                        else:
                            self.output_ref.transferDataTo(self.cloned_value)
                    return self.cloned_value
        else:
            import sys
            if MPI_DBG: print(process_rank, 'Waiting from ', self.remote_rank, 'with tag', self.tag, flush=True)
            output_data = process_comm.recv(source=self.remote_rank, tag=self.tag)
            if MPI_DBG: print('Received data from rank', self.remote_rank, 'with tag', self.tag, output_data, flush=True, file=sys.stderr)
            
            if MPI_DBG: print(process_rank, 'received successful obj type', type(output_data), flush=True)

            if type(output_data) is list:
                for v in output_data:
                    if v.xp_str == 'cp':
                        v.xp = cp
                    else:
                        v.xp = np
            else:
                if output_data.xp_str == 'cp':
                    output_data.xp = cp
                else:
                    output_data.xp = np

            if self.cloned_value is None:
                # TODO update copyTo to handle same target_device_idx but different rank
                if type(output_data) is list:
                    # if the output_ref is a list, we need to copy each element
                    self.cloned_value = [v.copyTo(target_device_idx) for v in output_data]
                else:
                    self.cloned_value = output_data.copyTo(target_device_idx)
                if MPI_DBG: print(process_rank, 'Received data copied', flush=True)
            else:
                # update transferDataTo to handle same target_device_idx but different rank
                if type(output_data) is list:
                    for output, cloned in zip(output_data, self.cloned_value):
                        output.transferDataTo(cloned)
                else:
                    output_data.transferDataTo(self.cloned_value)
                if MPI_DBG: print(process_rank, 'Received data transfered', flush=True)

            if MPI_DBG: print(process_rank, 'self.cloned_value', self.cloned_value)
            return self.cloned_value

    def set(self, value):
        if self.output_ref is not None:
            raise ValueError('InputValue already set, cannot set again')        
        if not isinstance(value, self.output_ref_type):
            raise ValueError(f'Value must be of type {self.output_ref_type} instead of {type(value)}')
        self.output_ref = value
    
    def type(self):
        return self.output_ref_type


class InputList():
    def __init__(self, type, optional=False):
        """
        Wrapper for input lists
        """
        self.output_ref_type = type
        self.input_values = []
        self.optional = optional

    def get(self, target_device_idx):
        return flatten([v.get(target_device_idx) for v in self.input_values])

    def set_list(self, other_list):
        """
        Set the input list with another list of values.
        If the input list is not empty, it will raise an error.
        """
        if len(self.input_values) > 0:
            raise ValueError('InputList already set, cannot set again')
        
        if not isinstance(other_list, list):
            raise ValueError(f'InputList must be set with a list, got {type(other_list)}')

        for v in other_list:
            self.input_values.append(InputValue(self.output_ref_type, optional=self.optional))
            self.input_values[-1].set(v)

    def set_item(self, item, index):
        while len(self.input_values) <= index:
            self.input_values.append(None)

        if self.input_values[index] is not None:
            raise ValueError(f'InputList item {index} already set, cannot set again')

        self.input_values[index] = InputValue(self.output_ref_type, optional=self.optional)
        if item is not None:
            self.input_values[index].set(item)
            
    def set_item_remote_rank(self, remote_rank, index):
        self.set_item(None, index)
        self.input_values[index].set_remote_rank(remote_rank)
    
    def set_item_tag(self, tag, index):
        self.input_values[index].set_tag(tag)

    def type(self):
        return self.output_ref_type
