
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue


sum_ = sum  # Preserve built-in


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
                 value2_remap: list=None,
                 target_device_idx: int=None,
                 precision: int=None,
                ):
        """
        Base operation processing object.

        Applies a sequence of element-wise operations to an input value (`value1`),
        optionally combining it with a second input (`value2`). The operation
        supports constant-based transformations as well as a single binary operation
        between two inputs.

        Parameters
        ----------
        constant_mul : float or array-like [1], optional
            Constant factor for element-wise multiplication.
        constant_div : float or array-like [1], optional
            Constant divisor for element-wise division.
        constant_sum : float or array-like [1], optional
            Constant added element-wise.
        constant_sub : float or array-like [1], optional
            Constant subtracted element-wise.
        constant_max : float or array-like [1], optional
            Element-wise lower bound (applies ``maximum(result, constant_max)``).
        constant_min : float or array-like [1], optional
            Element-wise upper bound (applies ``minimum(result, constant_min)``).
        mul : bool
            If True, multiply the result with ``value2``.
        div : bool
            If True, divide the result by ``value2``.
        sum : bool
            If True, add ``value2`` to the result.
        sub : bool
            If True, subtract ``value2`` from the result.
        concat : bool
            If True, concatenate ``value1`` and ``value2`` before applying
            constant operations.
        value2_remap : list of int, optional
            Optional index mapping applied to ``value2`` before the binary operation.
            Cannot be used together with ``concat``.
        target_device_idx : int [1], optional
            Target device index (CPU/GPU). If None, a global setting is used.
        precision : int [1], optional
            Precision for computation (0 = double, 1 = single). If None, a global
            setting is used.

        Raises
        ------
        ValueError
            If more than one of ``sum``, ``sub``, ``mul``, ``div`` or ``concat`` is True.
        ValueError
            If ``concat`` is True and ``value2_remap`` is provided.
        ValueError
            If a binary operation is requested but ``value2`` is not set during setup.

        Notes
        -----
        **Execution order**

        Operations are applied in the following order:

        1. Concatenation (if ``concat=True``)
        2. Constant multiplication and division
        3. Constant addition and subtraction
        4. Constant min/max (clamping)
        5. Binary operation with ``value2`` (if enabled)

        In pseudo-code::

            result = value1

            if concat:
                result = concat(result, value2)

            result *= constant_mul
            result /= constant_div

            result += constant_sum
            result -= constant_sub

            result = maximum(result, constant_max)
            result = minimum(result, constant_min)

            result = result (op) value2

        where ``(op)`` is one of ``+``, ``-``, ``*``, ``/``.

        **Broadcasting semantics**

        Constant operations are applied using in-place operators (e.g. ``*=``, ``+=``).
        As a result:

        - Broadcasting is supported only if it does not change the shape of the
        left-hand side (`value1`).
        - Shape-expanding broadcasts (e.g. from shape ``(1,)`` to ``(N,)``) are not
        allowed and will raise an exception.

        Examples::

            value shape (3,), constant scalar        → OK
            value shape (3,), constant shape (3,)    → OK
            value shape (1,), constant shape (3,)    → ERROR

        Binary operations with ``value2`` follow standard backend broadcasting rules
        (NumPy/CuPy).

        As an exception from standard broadcasting rule, if ``value2`` has a shorter first
        dimension than ``value1``, then the binary operation will only be applied to the first
        elements of ``value1``, while the rest will left untouched.

        **Constraints**

        - Only one binary operation flag can be active at a time.
        - ``value2`` must be provided when a binary operation or concatenation is used.
        - ``value2_remap`` cannot be used together with ``concat``.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if sum_([sum, sub, mul, div, concat]) > 1:
            raise ValueError('At most one of the "sum", "sub", "mul" "div" and "concat" flags can be set')

        if concat and value2_remap is not None:
            raise ValueError("value2_remap cannot be used with concatenation")

        # constant sum/sub and mul/div are combined together
        self.constant_sum = 0
        self.constant_mul = 1

        if constant_sum is not None:
            self.constant_sum += self.to_xp(self.xp.atleast_1d(constant_sum))
        if constant_sub is not None:
            self.constant_sum -= self.to_xp(self.xp.atleast_1d(constant_sub))

        if constant_mul is not None:
            self.constant_mul *= self.to_xp(self.xp.atleast_1d(constant_mul))
        if constant_div is not None:
            self.constant_mul /= self.to_xp(self.xp.atleast_1d(constant_div))

        # Max and min are treated separately
        if constant_max is not None:
            self.constant_max = self.to_xp(self.xp.atleast_1d(constant_max))
        else:
            self.constant_max = None

        if constant_min is not None:
            self.constant_min = self.to_xp(self.xp.atleast_1d(constant_min))
        else:
            self.constant_min = None

        self.mul = mul
        self.div = div
        self.sum = sum
        self.sub = sub
        self.concat = concat
        self.out_value = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.value2_remap = value2_remap

        self.inputs['in_value1'] = InputValue(type=BaseValue)
        self.inputs['in_value2'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_value'] = self.out_value

    @classmethod
    def input_names(cls):
        return {'in_value1': InputDesc(BaseValue, 'First input value vector'),
                'in_value2': InputDesc(BaseValue, 'Second input value vector (optional)')}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, 'Output value vector after applying the operation')}

    def setup(self):
        super().setup()

        value1 = self.local_inputs['in_value1']
        value2 = self.local_inputs['in_value2']

        # Check that both inputs have been set for operations that need them
        if self.mul or self.div or self.sum or self.sub or self.concat:
            if value2 is None:
                raise ValueError('in_value2 has not been set')

        # Allocate output value
        if self.concat:
            self.out_value.value = self.xp.empty(len(value1.value) + len(value2.value))
        else:
            self.out_value.value = self.xp.empty_like(value1.value)

        if value2 is not None:
            self.v2 = self.xp.zeros_like(value1.value)
            if self.mul or self.div:
                self.v2[:] = 1.0

    def trigger_code(self):

        value1 = self.local_inputs['in_value1'].value
        if self.local_inputs['in_value1'].generation_time < 0:
            value1 = self.xp.zeros_like(value1)
        if self.local_inputs['in_value2'] is not None:
            value2 = self.local_inputs['in_value2'].value
            if self.local_inputs['in_value2'].generation_time < 0:
                value2 = self.xp.zeros_like(value2)
        out = self.out_value.value

        if self.concat:
            v1_len = len(value1)
            out[:v1_len] = value1
            out[v1_len:] = value2
        else:
            out[:] = value1

        out *= self.constant_mul
        out += self.constant_sum

        if self.constant_max is not None:
            out[:] = self.xp.maximum(out, self.constant_max)

        if self.constant_min is not None:
            out[:] = self.xp.minimum(out, self.constant_min)

        if not self.concat and (self.sum or self.sub or self.mul or self.div):
            value2_is_shorter = len(value2) < len(value1)

            if value2_is_shorter:
                self.v2[:len(value2)] = value2
            elif self.value2_remap is not None:
                self.v2[self.value2_remap] = value2
            else:
                self.v2 = value2  # Move reference

            if self.mul:
                out *= self.v2
            elif self.div:
                out /= self.v2
            elif self.sum:
                out += self.v2
            elif self.sub:
                out -= self.v2

        self.out_value.generation_time = self.current_time
