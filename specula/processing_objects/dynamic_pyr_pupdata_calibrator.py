

from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.processing_objects.pyr_pupdata_calibrator import PyrPupdataCalibrator


class DynamicPyrPupdataCalibrator(PyrPupdataCalibrator):
    """Dynamic Pyramid Pupdata Calibrator.

    A version of PyrPupdataCalibrator that
    can save its output when receiving a trigger on the 'in_save' input.
    """
    

    def __init__(self,
                 data_dir: str,      # Set by main Simul object
                 dt: float = None,
                 thr1: float = 0.1,
                 thr2: float = 0.25,
                 obs_thr: float = 0.8,
                 slopes_from_intensity: bool=False,
                 output_tag: str = None,
                 auto_detect_obstruction: bool = True,
                 min_obstruction_ratio: float = 0.05,
                 display_debug: bool = False,
                 overwrite: bool = False,
                 save_on_exit: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(data_dir=data_dir, dt=dt, thr1=thr1, thr2=thr2, obs_thr=obs_thr,
                         slopes_from_intensity=slopes_from_intensity, output_tag=output_tag,
                         auto_detect_obstruction=auto_detect_obstruction,
                         min_obstruction_ratio=min_obstruction_ratio, display_debug=display_debug,
                         overwrite=overwrite, save_on_exit=save_on_exit,
                         target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_save'] = InputValue(type=BaseValue, optional=True)

    def post_trigger(self):
        super().post_trigger()

        input_save = self.local_inputs['in_save']
        if input_save is not None and input_save.generation_time == self.current_time:
            self._save()

