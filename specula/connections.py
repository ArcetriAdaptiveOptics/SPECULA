from specula import process_rank, process_comm, MPI_DBG, MPI_SEND_DBG
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

    def get(self, target_device_idx):
        if not self.remote:
            if self.output_ref is None:
                return None
            if self.output_ref.target_device_idx == target_device_idx:
                return self.output_ref

        if not self.remote:            
            value_to_copy = self.output_ref
        else:
            value_to_copy = process_comm.recv(source=self.remote_rank, tag=self.tag)
            for v in value_to_copy if type(value_to_copy) is list else [value_to_copy]:
                if v.xp_str == 'cp':
                    v.xp = cp
                else:
                    v.xp = np

        if self.cloned_value is None:
            if type(value_to_copy) is list:
                self.cloned_value = [v.copyTo(target_device_idx) for v in value_to_copy]
            else:
                self.cloned_value = value_to_copy.copyTo(target_device_idx)
        else:
            if type(value_to_copy) is list:
                for output, cloned in zip(value_to_copy, self.cloned_value):
                    output.transferDataTo(cloned)
            else:
                value_to_copy.transferDataTo(self.cloned_value)
        return self.cloned_value

    def set(self, value):
        if self.output_ref is not None:
            raise ValueError('InputValue already set, cannot set again')        
        if not isinstance(value, self.output_ref_type):
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

