from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray, np


class DataPrint(BaseProcessingObj):
    '''Print data values to screen at regular intervals'''

    def __init__(self,
                 print_dt: float = 1.0,      # Print interval in seconds
                 range_slice: tuple = None,      # Range of values to print (e.g., (0, 5))
                 prefix: str = '',               # Prefix for printed output
                 format_str: str = '.4f',        # Format string for numbers
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Initialize the data print object.

        Parameters:
        print_dt (float): Time interval between prints in seconds
        range_slice (tuple, optional): Tuple to create slice object to select which values to print.
                                       If None, prints all values.
                                       Examples: (0, 5), (None, None, 2)
        prefix (str): Text to print before the values
        format_str (str): Format specification for floating point numbers
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.print_dt = self.seconds_to_t(print_dt)
        self.range_slice = slice(*range_slice) if range_slice is not None else slice(None)
        self.prefix = prefix
        self.format_str = format_str
        self.last_print_time = -self.print_dt  # Print on first trigger

        self.inputs['in_value'] = InputValue(type=BaseValue)

    def trigger(self):
        # Check if it's time to print
        if (self.current_time - self.last_print_time) >= self.print_dt:
            value = self.local_inputs['in_value'].value
            # Transfer to CPU if needed
            value_cpu = cpuArray(value)

            # Check if value is None
            if value_cpu is None:
                print(f"t={self.current_time_seconds:.4f}s {self.prefix}: None")
                self.last_print_time = self.current_time
                return

            # Ensure it's a numpy array
            if not isinstance(value_cpu, np.ndarray):
                value_cpu = np.asarray(value_cpu)

            # Select range (only if array has dimensions)
            if value_cpu.ndim > 0:
                selected = value_cpu[self.range_slice]
            else:
                selected = value_cpu

            # Check if selected is None after slicing
            if selected is None:
                print(f"t={self.current_time_seconds:.4f}s {self.prefix}: None (after slicing)")
                self.last_print_time = self.current_time
                return

            # Format output
            time_str = f"t={self.current_time_seconds:.4f}s"
            if self.prefix:
                output = f"{time_str} {self.prefix}: "
            else:
                output = f"{time_str}: "

            # Format values
            if selected.ndim == 0:  # Scalar
                try:
                    scalar_val = float(selected.item() if hasattr(selected, 'item') else selected)
                    output += f"{scalar_val:{self.format_str}}"
                except (TypeError, ValueError):
                    output += str(selected)
            elif selected.ndim == 1:  # 1D array
                values_str = ", ".join([f"{v:{self.format_str}}" for v in selected])
                output += f"[{values_str}]"
            else:  # 2D or higher
                output += f"shape={selected.shape}, "
                flat = selected.flatten()[:10]  # Print first 10 elements
                values_str = ", ".join([f"{v:{self.format_str}}" for v in flat])
                output += f"[{values_str}{'...' if selected.size > 10 else ''}]"

            print(output)

            self.last_print_time = self.current_time