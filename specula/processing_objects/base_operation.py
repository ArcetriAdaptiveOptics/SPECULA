from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


class BaseOperation(BaseProcessingObj):
    ''''Simple operations with base value(s)'''

    def __init__(self, constant_mult=None, constant_div=None, constant_sum=None, constant_sub=None, mult=False, div=False, sum=False, sub=False,
                 target_device_idx=None, precision=None):
        """
        Initialize the base operation object.

        Parameters:
        constant_mult (float, optional): Constant for multiplication
        constant_div (float, optional): Constant for division
        constant_sum (float, optional): Constant for addition
        constant_sub (float, optional): Constant for subtraction
        mult (bool, optional): Flag for multiplication operation
        div (bool, optional): Flag for division operation
        sum (bool, optional): Flag for addition operation
        sub (bool, optional): Flag for subtraction operation
        """
        super().__init__()

        self._constant_mult = constant_mult
        self._constant_sum = constant_sum
        if constant_div is not None:
            self._constant_mult = 1.0 / constant_div
        if constant_sub is not None:
            self._constant_sum = -constant_sub

        self._mult = mult
        self._div = div
        self._sum = sum
        self._sub = sub
        self._out_value = BaseValue(target_device_idx=target_device_idx)

        self.inputs['in_value1'] = InputValue(type=BaseValue)
        self.inputs['in_value2'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self._out_value

    @property
    def in_value1(self):
        return self._in_value1

    @in_value1.setter
    def in_value1(self, value):
        self._in_value1 = value

    @property
    def in_value2(self):
        return self._in_value2

    @in_value2.setter
    def in_value2(self, value):
        self._in_value2 = value

    @property
    def out_value(self):
        return self._out_value

    def trigger_code(self):
        value1 = self.inputs['in_value1'].get(self.target_device_idx)
        value2 = self.inputs['in_value2'].get(self.target_device_idx)
        if value1 and value1.generation_time == self.current_time:
            if self._constant_mult is not None:
                self._out_value.value = value1.value * self._constant_mult
            if self._constant_sum is not None:
                self._out_value.value = value1.value + self._constant_sum
            self._out_value.generation_time = self.current_time

        if value1 and value2 and (value1.generation_time == self.current_time or value2.generation_time ==  self.current_time):
            temp = 0
            if self._mult:
                if value1.value is not None: temp = value1.value.copy()
                if value2.value is not None: temp *= value2.value.copy()
            if self._div:
                if value1.value is not None: temp = value1.value.copy()
                if value2.value is not None: temp /= value2.value.copy()
            if self._sum:
                if value1.value is not None: temp = value1.value.copy()
                if value2.value is not None: temp += value2.value.copy()
            if self._sub:
                if value1.value is not None: temp = value1.value.copy()
                if value2.value is not None: temp -= value2.value.copy()
            self._out_value.value = temp
            self._out_value.generation_time = self.current_time

    def run_check(self, time_step, errmsg=None):
        """
        Check the validity of the operation.

        Parameters:
        time_step (int): The time step for the simulation
        errmsg (str, optional): Error message

        Returns:
        bool: True if the check is successful, False otherwise
        """
        return self._out_value is not None and self.inputs['in_value1'].get(self.target_device_idx) is not None

    def save(self, filename):
        hdr = fits.Header()
        hdr['CONST_MULT'] = self._constant_mult
        hdr['CONST_DIV'] = self._constant_div
        hdr['CONST_SUM'] = self._constant_sum
        hdr['CONST_SUB'] = self._constant_sub
        hdr['MULT'] = self._mult
        hdr['DIV'] = self._div
        hdr['SUM'] = self._sum
        hdr['SUB'] = self._sub
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['CONST_MULT'] = self._constant_mult
            hdr['CONST_DIV'] = self._constant_div
            hdr['CONST_SUM'] = self._constant_sum
            hdr['CONST_SUB'] = self._constant_sub
            hdr['MULT'] = self._mult
            hdr['DIV'] = self._div
            hdr['SUM'] = self._sum
            hdr['SUB'] = self._sub
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._constant_mult = hdr.get('CONST_MULT', None)
            self._constant_div = hdr.get('CONST_DIV', None)
            self._constant_sum = hdr.get('CONST_SUM', None)
            self._constant_sub = hdr.get('CONST_SUB', None)
            self._mult = hdr.get('MULT', False)
            self._div = hdr.get('DIV', False)
            self._sum = hdr.get('SUM', False)
            self._sub = hdr.get('SUB', False)

