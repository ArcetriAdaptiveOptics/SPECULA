
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField


class PhaseExtractor(BaseProcessingObj):
    """
    Phase extractor processing object.
    Extracts the phase (phaseInNm) from an ElectricField or Layer and stores it
    as a BaseValue, preserving the original 2-D shape.
    """
    def __init__(self,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Phase extractor processing object.

        Reads an ElectricField (or Layer, which is a subclass) and copies its
        ``phaseInNm`` 2-D array into a ``BaseValue`` output, preserving the original shape.

        Parameters
        ----------
        target_device_idx : int [1], optional
            Target device index (CPU/GPU). If None, a global setting is used.
        precision : int [1], optional
            Precision (0 = double, 1 = single). If None, a global setting is used.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.out_phase = BaseValue(target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_phase'] = self.out_phase

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field or layer')}

    @classmethod
    def output_names(cls):
        return {'out_phase': OutputDesc(BaseValue, 'Phase extracted from the input field [nm]')}

    def setup(self):
        super().setup()
        ef = self.local_inputs['in_ef']
        self.out_phase.value = self.xp.empty(ef.phaseInNm.shape, dtype=self.dtype)

    def trigger_code(self):
        ef = self.local_inputs['in_ef']
        self.out_phase.value[:] = ef.phaseInNm
        self.out_phase.generation_time = self.current_time
