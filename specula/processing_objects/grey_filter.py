from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.intensity import Intensity


class GreyFilter(BaseProcessingObj):
    """
    Grey (neutral density) filter: the output intensity is the input intensity
    times a transmission in [0, 1].

    Placed between a wavefront sensor and its detector (e.g. ``sh.out_i`` ->
    ``detector.in_i``), it emulates a fainter source: the detector then adds
    photon, readout and excess noise to the attenuated flux. A transmission of
    10**(-0.4 * dm) corresponds to a source fainter by dm magnitudes.

    The transmission is the ``transmission`` parameter, or, if the optional
    ``in_transmission`` input is connected (e.g. to a generator), its value at
    every step. The value of the input is not checked at run time.
    """

    def __init__(self,
                 transmission: float = 1.0,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters
        ----------
        transmission : float, optional
            Transmission in [0, 1], used when ``in_transmission`` is not
            connected (default: 1.0).
        target_device_idx : int, optional
            Target device for computation (-1 for CPU, >=0 for GPU).
        precision : int, optional
            Numerical precision (0 for double, 1 for single).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        if not 0.0 <= transmission <= 1.0:
            raise ValueError(f'transmission must be in [0, 1], got {transmission}')
        self.transmission = transmission
        # Placeholder size, reallocated in setup() from the input
        self.out_i = Intensity(1, 1, target_device_idx=self.target_device_idx, precision=precision)
        self.inputs['in_i'] = InputValue(type=Intensity)
        self.inputs['in_transmission'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_i'] = self.out_i

    @classmethod
    def input_names(cls):
        return {'in_i': InputDesc(Intensity, 'Input intensity'),
                'in_transmission': InputDesc(BaseValue, 'Transmission in [0, 1], one value '
                                                        '(optional, replaces the parameter)')}

    @classmethod
    def output_names(cls):
        return {'out_i': OutputDesc(Intensity, 'Input intensity times the transmission')}

    def setup(self):
        super().setup()
        self.out_i.i = self.xp.zeros(self.local_inputs['in_i'].i.shape, dtype=self.dtype)

    def trigger_code(self):
        in_i = self.local_inputs['in_i'].i
        in_transmission = self.local_inputs['in_transmission']
        if in_transmission is None:
            self.xp.multiply(in_i, self.transmission, out=self.out_i.i)
        else:
            # One-element array, broadcast on the device: no host sync
            self.xp.multiply(in_i, in_transmission.value, out=self.out_i.i)

    def post_trigger(self):
        super().post_trigger()
        self.out_i.generation_time = self.current_time
