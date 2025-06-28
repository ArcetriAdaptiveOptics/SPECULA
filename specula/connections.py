from specula import process_rank, process_comm, MPI_DBG
from specula import np, cp

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
                        self.cloned_value = self.output_ref.copyTo(target_device_idx)
                    else:
                        self.output_ref.transferDataTo(self.cloned_value)
                    return self.cloned_value
        else:
            if MPI_DBG: print(process_rank, 'Receiving from ', self.remote_rank, 'with tag', self.tag, flush=True)
            output_data = process_comm.recv(source=self.remote_rank, tag=self.tag)
            if output_data.xp_str == 'cp':
                output_data.xp = cp
            else:
                output_data.xp = np
            
            if MPI_DBG: print(process_rank, 'Receive successful:', output_data, flush=True)
            if MPI_DBG: print(process_rank, 'received obj type', type(output_data), flush=True)
            if MPI_DBG: print(process_rank, 'received obj', output_data, flush=True)

            if self.cloned_value is None:
                # update copyTo to handle same target_device_idx but different rank
                self.cloned_value = output_data.copyTo(target_device_idx)
                if MPI_DBG: print(process_rank, 'Received data copied', flush=True)
            else:
                # update transferDataTo to handle same target_device_idx but different rank
                output_data.transferDataTo(self.cloned_value)
                if MPI_DBG: print(process_rank, 'Received data transfered', flush=True)

            if MPI_DBG: print(process_rank, 'self.cloned_value', self.cloned_value)
            return self.cloned_value

    def set(self, value):
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
        self.output_ref_list = None
        self.cloned_list = []
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
        if not self.output_ref_list is None:
            return [x.generation_time for x in self.output_ref_list]
        else:
            return []

    def get(self, target_device_idx):
        '''Copy all values in the list to the specified target'''
        if self.output_ref_list is None:
            return

        if not self.remote:
            if self.cloned_list == []:
                # First get(): allocate another object with copyTo where needed
                for list_item in self.output_ref_list:
                        if list_item.target_device_idx == target_device_idx:
                            self.cloned_list.append(list_item)
                        else:
                            self.cloned_list.append(list_item.copyTo(target_device_idx))             
            else:
                # Second get(): always used transferDataTo()            
                for i, (list_item, cloned) in enumerate(zip(self.output_ref_list, self.cloned_list)):                    
                    if list_item.target_device_idx == target_device_idx:
                        self.cloned_list[i] = list_item
                    else:
                        list_item.transferDataTo(cloned)
        else:
            if MPI_DBG: print(process_rank, 'Receiveing List from ', self.remote_rank, 'with tag', self.tag, flush=True)
            output_data_list = process_comm.recv(source=self.remote_rank, tag=self.tag)
            if MPI_DBG: print(process_rank, 'Receive List successful:', output_data_list, flush=True)
            #if self.cloned_list == []:
                # First get(): allocate another object with copyTo where needed
            for list_item in output_data_list:                                        
                self.cloned_list.append(list_item.copyTo(target_device_idx))                        
            # else:
            #    # Second get(): always used transferDataTo()            
            #    for i, (list_item, cloned) in enumerate(zip(self.output_ref_list, output_data_list)):
            #        list_item.transferDataTo(cloned)

        return self.cloned_list

    def set(self, new_list):
        for value in new_list:
            if not isinstance(value, self.output_ref_type):
                raise ValueError(f'List element must be of type {self.output_ref_type}')
        self.output_ref_list = new_list

    def type(self):
        return self.output_ref_type
