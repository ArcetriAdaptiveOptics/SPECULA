

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
        self.inputs['in_dt'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_thr1'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_thr2'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_output_tag'] = InputValue(type=BaseValue, optional=True)

        self.outputs['out_params'] = BaseValue()

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Interactive inputs are protected from exceptions
        try:
            # Use float() to accept string values as well
            input_dt = self.local_inputs['in_dt']
            if input_dt is not None and input_dt.generation_time == self.current_time:
                self.dt = self.seconds_to_t(float(input_dt.value))
        
            input_thr1 = self.local_inputs['in_thr1']
            if input_thr1 is not None and input_thr1.generation_time == self.current_time:
                self.thr1 = float(input_thr1.value)

            input_thr2 = self.local_inputs['in_thr2']
            if input_thr2 is not None and input_thr2.generation_time == self.current_time:
                self.thr2 = float(input_thr2.value)
        except Exception as e:
            print(f'Exception: {e.__name__}: {e}')

    def trigger_code(self):

        try:
            super().trigger_code()
            self.status_string = 'OK'
        except (ValueError, TypeError) as e:
            # Skip iterations in case of errors
            self.status_string = f'{e.__class__.__name__}: {e}'

    def post_trigger(self):
        super().post_trigger()

        # Save pupdata if requested
        input_save = self.local_inputs['in_save']
        if input_save is not None and input_save.generation_time == self.current_time:
            self._save(input_save.value)

        # Update output params with current values
        self.outputs['out_params'].value = {
            'dt': self.t_to_seconds(self.dt),
            'thr1': self.thr1,
            'thr2': self.thr2,
            'status': self.status_string,
        }

        self.outputs['out_params'].generation_time = self.current_time




