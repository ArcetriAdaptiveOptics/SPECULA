
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
        self.cloned_list = []
        self.optional = optional

    def get_time(self):
        if not self.wrapped_list is None:
            return [x.generation_time for x in self.wrapped_list]
        else:
            return []

    def get(self, target_device_idx):
        '''Copy all values in the list to the specified target'''
        if self.wrapped_list is None:
            return

        if self.cloned_list == []:
            # First get(): allocate another object with copyTo where needed
            for wrapped in self.wrapped_list:
                if wrapped.target_device_idx == target_device_idx:
                    self.cloned_list.append(wrapped)
                else:
                    self.cloned_list.append(wrapped.copyTo(target_device_idx))
        else:
            # Second get(): alwats used transferDataTo()
            for i, (wrapped, cloned) in enumerate(zip(self.wrapped_list, self.cloned_list)):
                if wrapped.target_device_idx == target_device_idx:
                    self.cloned_list[i] = wrapped
                else:
                    wrapped.transferDataTo(cloned)
        return self.cloned_list

    def set(self, new_list):
        for value in new_list:
            if not isinstance(value, self.wrapped_type):
                raise ValueError(f'List element must be of type {self.wrapped_type}')
        self.wrapped_list = new_list

    def type(self):
        return self.wrapped_type
