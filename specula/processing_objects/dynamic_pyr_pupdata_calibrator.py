

from specula.base_value import BaseValue
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.connections import InputValue
from specula.data_objects.intensity import Intensity
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.processing_objects.pyr_pupdata_calibrator import PyrPupdataCalibrator


class DynamicPyrPupdataCalibrator(PyrPupdataCalibrator):
    """
    Dynamic Pyramid Pupdata Calibrator. Pyramid pupil data calibrator with interactive control.

    This class extends :class:`PyrPupdataCalibrator` by adding support for
    dynamic parameter updates and on-demand saving via input triggers.
    Calibration parameters such as time step and thresholds can be modified
    during runtime without restarting the processing pipeline.

    A failed pupil measurement will not stop the simulation, but will be instead signalled
    using the "status_string" member of the "out_params" dictionary, as well by
    the fact that the output pudata will not be refreshed.
    Whenever a pupil measurement is successful, status_string will be set to "OK".

    Interactive inputs
    ------------------
    in_dt : BaseValue, optional
        Dynamically update the time step (seconds).
    in_thr1 : BaseValue, optional
        Dynamically update the first threshold.
    in_thr2 : BaseValue, optional
        Dynamically update the second threshold.
    in_output_tag : BaseValue, optional
        Dynamically update the output tag.
    in_save : BaseValue, optional
        Trigger to save the current calibration data.

    Outputs
    -------
    out_params : BaseValue
        Dictionary containing current calibration parameters and status:
        ``{'dt': float, 'thr1': float, 'thr2': float, 'status': str}``.
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
        """
        Parameters
        ----------
        data_dir : str
            Directory where calibration data is stored.
        dt : float, optional
            Time step for processing (in seconds).
        thr1 : float, optional
            First threshold used in pupil processing. Default is 0.1.
        thr2 : float, optional
            Second threshold used in pupil processing. Default is 0.25.
        obs_thr : float, optional
            Threshold for obstruction detection. Default is 0.8.
        slopes_from_intensity : bool, optional
            If True, compute indices suitable for calculation of slopes from intensity. Default is False.
        output_tag : str, optional
            Tag used to label output files.
        auto_detect_obstruction : bool, optional
            Enable automatic obstruction detection. Default is True.
        min_obstruction_ratio : float, optional
            Minimum obstruction ratio to consider. Default is 0.05.
        display_debug : bool, optional
            If True, enable debug visualization. Default is False.
        overwrite : bool, optional
            If True, overwrite existing files. Default is False.
        save_on_exit : bool, optional
            If True, automatically save data on exit. Default is True.
        target_device_idx : int, optional
            Target device index for computation.
        precision : int, optional
            Numerical precision for internal data (0 for double, 1 for single).
        """    
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

    @classmethod
    def input_names(cls):
        return {'in_i': InputDesc(Intensity, 'Input intensity from the pyramid WFS detector (optional)'),
                'in_pixels': InputDesc(Pixels, 'Input pixel data from the detector (optional)'),
                'in_save': InputDesc(BaseValue, 'Trigger to save the current calibration data (optional)'),
                'in_dt': InputDesc(BaseValue, 'Dynamically update the time step in seconds (optional)'),
                'in_thr1': InputDesc(BaseValue, 'Dynamically update the first threshold (optional)'),
                'in_thr2': InputDesc(BaseValue, 'Dynamically update the second threshold (optional)'),
                'in_output_tag': InputDesc(BaseValue, 'Dynamically update the output tag (optional)')}

    @classmethod
    def output_names(cls):
        return {'out_pupdata': OutputDesc(PupData, 'Calibrated pupil data with subaperture geometry'),
                'out_params': OutputDesc(BaseValue, 'Dictionary with current calibration parameters and status')}

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Interactive inputs are protected from exceptions
        try:
            # Use float() to accept string values as well
            input_dt = self.local_inputs['in_dt']
            if input_dt is not None and input_dt.generation_time == self.current_time:
                self.dt = self.seconds_to_t(float(input_dt.value))
        except Exception as e:
            print(f'Exception: {e.__name__}: {e}')

        try:
            input_thr1 = self.local_inputs['in_thr1']
            if input_thr1 is not None and input_thr1.generation_time == self.current_time:
                self.thr1 = float(input_thr1.value)
        except Exception as e:
            print(f'Exception: {e.__name__}: {e}')

        try:
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




