
class InputValue():
    def __init__(self, type, optional=False):
        """
        Wrapper for simple input values
        """
        self.wrapped_type = type
        self.wrapped_value = None
        self.cloned_value = None
        self.optional = optional

    def get_time(self):
        if not self.wrapped_value is None:
            return self.wrapped_value.generation_time        

    def get(self, target_device_idx):
        if not self.wrapped_value is None:
            if self.wrapped_value.target_device_idx == target_device_idx:
                return self.wrapped_value
            else:
                if self.cloned_value is None:
                    self.cloned_value = self.wrapped_value.copyTo(target_device_idx)
                else:
                    self.wrapped_value.transferDataTo(self.cloned_value)
                return self.cloned_value

    def set(self, value):
        if not isinstance(value, self.wrapped_type):
            raise ValueError(f'Value must be of type {self.wrapped_type} instead of {type(value)}')
        self.wrapped_value = value
    
    def type(self):
        return self.wrapped_type


class InputList():
    def __init__(self, type, optional=False):
        """
        Wrapper for input lists
        """
        self.wrapped_type = type
        self.wrapped_list = None
        self.optional = optional

    def get_time(self):
        if not self.wrapped_list is None:
            return [x.generation_time for x in self.wrapped_list]
        else:
            return []

    def get(self, target_device_idx):
        '''Copy all values in the list to the specified target

        TODO this should use transferTo() when possible like InputValue'''
        if not self.wrapped_list is None:            
            return [x.copyTo(target_device_idx) for x in self.wrapped_list]

    def set(self, new_list):
        for value in new_list:
            if not isinstance(value, self.wrapped_type):
                raise ValueError(f'List element must be of type {self.wrapped_type}')
        self.wrapped_list = new_list

    def type(self):
        return self.wrapped_type
