
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


class BaseOperation(BaseProcessingObj):
    """
    Base Operation processing object.
    Simple operations with base value(s).
    """
    def __init__(self,
                 constant_mul: float=None,
                 constant_div: float=None,
                 constant_sum: float=None,
                 constant_sub: float=None,
                 constant_max: float=None,
                 constant_min: float=None,
                 mul: bool=False,
                 div: bool=False,
                 sum: bool=False,
                 sub: bool=False,
                 concat: bool=False,
                 value2_is_shorter: bool=False,
                 value2_remap: list=None,
                 target_device_idx: int=None,
                 precision:int =None):
        """
        Initialize the base operation object.

        Parameters:
        constant_mul (float, optional): Constant for multiplication
        constant_div (float, optional): Constant for division
        constant_sum (float, optional): Constant for addition
        constant_sub (float, optional): Constant for subtraction
        constant_max (float, optional): Constant for maximum
        constant_min (float, optional): Constant for minimum
        mul (bool, optional): Flag for multiplication operation
        div (bool, optional): Flag for division operation
        sum (bool, optional): Flag for addition operation
        sub (bool, optional): Flag for subtraction operation
        concat (bool, optional): Flag for concatenation operation
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Implement constant div and sub as reciprocal of mul and sum
        if not constant_mul is None:
            if self.xp.isscalar(constant_mul):
                self.constant_mul = constant_mul
            else:
                self.constant_mul = self.to_xp(constant_mul)
        else:
            self.constant_mul = None
        if not constant_sum is None:
            if self.xp.isscalar(constant_sum):
                self.constant_sum = constant_sum
            else:
                self.constant_sum = self.to_xp(constant_sum)
        else:
            self.constant_sum = None
        if not constant_div is None:
            if self.xp.isscalar(constant_div):
                self.constant_mul = 1.0 / constant_div
            else:
                self.constant_mul = 1.0 / self.to_xp(constant_div)
        if not constant_sub is None:
            if self.xp.isscalar(constant_sub):
                self.constant_sum = -constant_sub
            else:
                self.constant_sum = -self.to_xp(constant_sub)

        if not constant_max is None:
            if self.xp.isscalar(constant_max):
                self.constant_max = constant_max
            else:
                self.constant_max = self.xp.max(constant_max)
        else:
            self.constant_max = None
        if not constant_min is None:
            if self.xp.isscalar(constant_min):
                self.constant_min = constant_min
            else:
                self.constant_min = self.xp.min(constant_min)
        else:
            self.constant_min = None

        self.mul = mul
        self.div = div
        self.sum = sum
        self.sub = sub
        self.concat = concat
        self.out_value = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.value2_is_shorter = value2_is_shorter
        self.value2_remap = value2_remap

        self.inputs['in_value1'] = InputValue(type=BaseValue)
        self.inputs['in_value2'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_value'] = self.out_value

    def setup(self):
        super().setup()

        value1 = self.local_inputs['in_value1']
        value2 = self.local_inputs['in_value2']

        # Check that both inputs have been set for
        # operations that need them
        if self.mul or self.div or self.sum or self.sub or self.concat:
            if value2 is None:
                raise ValueError('in_value2 has not been set')

        # Allocate output value
        if not self.constant_mul is None or not self.constant_sum is None:
            self.out_value.value = value1.value * 0.0
        elif not self.constant_max is None:
            self.out_value.value = self.constant_max
        elif not self.constant_min is None:
            self.out_value.value = self.constant_min
        elif self.concat:
            self.out_value.value = self.xp.empty(len(value1.value) + len(value2.value))
        else:
            self.out_value.value = self.xp.empty_like(value1.value)

        if value2 is not None:
            self.v2 = self.xp.empty_like(value1.value)
            if self.div:
                self.v2[:] = 1.0
            else:
                self.v2[:] = 0.0

    def trigger_code(self):

        value1 = self.local_inputs['in_value1'].value

        if not self.constant_mul is None:
            self.out_value.value[:] = value1 * self.constant_mul

        elif not self.constant_sum is None:
            self.out_value.value[:] = value1 + self.constant_sum

        elif not self.constant_max is None:
            self.out_value.value = self.xp.maximum(value1,self.constant_max)

        elif not self.constant_min is None:
            self.out_value.value = self.xp.minimum(value1,self.constant_min)

        else:
            value2 = self.local_inputs['in_value2'].value

            out = self.out_value.value
            if self.concat:
                out[:len(value1)] = value1
                out[len(value1):] = value2
            else:
                if self.value2_is_shorter:
                    self.v2[:len(value2)] = value2
                elif self.value2_remap is not None:
                    self.v2[self.value2_remap] = value2
                else:
                    self.v2[:] = value2

                if self.mul:
                    out[:] = value1 * self.v2
                elif self.div:
                    out[:] = value1 / self.v2
                elif self.sum:
                    out[:] = value1 + self.v2
                elif self.sub:
                    out[:] = value1 - self.v2
                else:
                    raise ValueError('No operation defined')

        self.out_value.generation_time = self.current_time
