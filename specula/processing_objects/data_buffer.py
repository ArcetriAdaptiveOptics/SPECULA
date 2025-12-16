
from collections import OrderedDict, defaultdict

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue


class DataBuffer(BaseProcessingObj):
    '''Data buffering object - accumulates data and outputs it every N steps'''

    def __init__(self, buffer_size: int = 10):
        super().__init__()
        self.buffer_size = buffer_size
        self.storage = defaultdict(OrderedDict)
        self.step_counter = 0
        self.buffered_outputs = {}

    def setOutputs(self):
        # Create output objects for each input (like DataStore does)
        for input_name, input_obj in self.inputs.items():
            if input_obj is not None:
                # Create output name and object
                output_name = f"{input_name}_buffered"
                output_obj = BaseValue(target_device_idx=self.target_device_idx)
                self.buffered_outputs[output_name] = output_obj
                self.outputs[output_name] = output_obj

    def trigger_code(self):
        # Accumulate data (same logic as DataStore)
        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                v = item.get_value()
                self.storage[k][self.step_counter] = v.copy()

        self.step_counter += 1

        if self.step_counter >= self.buffer_size:
            self.emit_buffered_data()
            self.reset_buffers()

    def emit_buffered_data(self):
        for input_name, data_dict in self.storage.items():
            if len(data_dict) == 0:
                continue
            output_name = f"{input_name}_buffered"
            values = self.xp.array(list(data_dict.values()))
            if output_name in self.buffered_outputs:
                self.buffered_outputs[output_name].value = values
                self.buffered_outputs[output_name].generation_time = self.current_time
                if self.verbose:
                    print(f"DataBuffer: emitted {len(values)} samples for {input_name}")

    def setup(self):
        # We check that all input items
        for k, _input in self.inputs.items():
            item = _input.get(target_device_idx=self.target_device_idx)
            if item is not None and not hasattr(item, 'get_value'):
                raise TypeError(f"Error: don't know how to buffer an object of type {type(item)}")                

    def reset_buffers(self):
        """Clear all buffers and reset counter"""
        self.storage.clear()
        self.step_counter = 0

        if self.verbose:
            print(f"DataBuffer: reset buffers at time {self.current_time}")

    def finalize(self):
        """Emit any remaining data in buffers"""        
        if self.step_counter > 0:
            self.emit_buffered_data()
            self.reset_buffers()
        
        super().finalize()