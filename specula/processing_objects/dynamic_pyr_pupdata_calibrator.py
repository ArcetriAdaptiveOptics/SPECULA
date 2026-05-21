

from specula.base_value import BaseValue
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.connections import InputValue
from specula.processing_objects.pyr_pupdata_calibrator import PyrPupdataCalibrator
from specula.scalar_values import IntValue, StringValue, FloatValue


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
        dt : float [s], optional
            Time step for processing (in seconds).
        thr1 : float [1], optional
            First threshold used in pupil processing. Default is 0.1.
        thr2 : float [1], optional
            Second threshold used in pupil processing. Default is 0.25.
        obs_thr : float [1], optional
            Threshold for obstruction detection. Default is 0.8.
        slopes_from_intensity : bool
            If True, compute indices suitable for calculation of slopes from intensity. Default is False.
        output_tag : str
            Tag used to label output files.
        auto_detect_obstruction : bool
            Enable automatic obstruction detection. Default is True.
        min_obstruction_ratio : float [1], optional
            Minimum obstruction ratio to consider. Default is 0.05.
        display_debug : bool
            If True, enable debug visualization. Default is False.
        overwrite : bool
            If True, overwrite existing files. Default is False.
        save_on_exit : bool
            If True, automatically save data on exit. Default is True.
        target_device_idx : int [1], optional
            Target device index for computation.
        precision : int [1], optional
            Numerical precision for internal data (0 for double, 1 for single).
        """    
        super().__init__(data_dir=data_dir, dt=dt, thr1=thr1, thr2=thr2, obs_thr=obs_thr,
                         slopes_from_intensity=slopes_from_intensity, output_tag=output_tag,
                         auto_detect_obstruction=auto_detect_obstruction,
                         min_obstruction_ratio=min_obstruction_ratio, display_debug=display_debug,
                         overwrite=overwrite, save_on_exit=save_on_exit,
                         target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_save'] = InputValue(type=IntValue, optional=True)
        self.inputs['in_dt'] = InputValue(type=FloatValue, optional=True)
        self.inputs['in_thr1'] = InputValue(type=FloatValue, optional=True)
        self.inputs['in_thr2'] = InputValue(type=FloatValue, optional=True)
        self.inputs['in_output_tag'] = InputValue(type=StringValue, optional=True)

        self.outputs['out_params'] = StringValue("")

    @classmethod
    def input_names(cls):
        result = super().input_names()
        result.update({
                'in_save': InputDesc(IntValue, 'Trigger to save the current calibration data (optional)'),
                'in_dt': InputDesc(FloatValue, 'Dynamically update the time step in seconds (optional)'),
                'in_thr1': InputDesc(FloatValue, 'Dynamically update the first threshold (optional)'),
                'in_thr2': InputDesc(FloatValue, 'Dynamically update the second threshold (optional)'),
                'in_output_tag': InputDesc(StringValue, 'Dynamically update the output tag (optional)')
                })
        return result

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({
                'out_params': OutputDesc(StringValue, 'Dictionary with current calibration parameters and status')
                })
        return result

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        input_dt = self.local_inputs['in_dt']
        if input_dt is not None and input_dt.generation_time == self.current_time:
            self.dt = self.seconds_to_t(input_dt.value)

        input_thr1 = self.local_inputs['in_thr1']
        if input_thr1 is not None and input_thr1.generation_time == self.current_time:
            self.thr1 = input_thr1.value

        input_thr2 = self.local_inputs['in_thr2']
        if input_thr2 is not None and input_thr2.generation_time == self.current_time:
            self.thr2 = input_thr2.value

        input_tag = self.local_inputs['in_output_tag']
        if input_tag is not None and input_tag.generation_time == self.current_time:
            self.filename = input_tag.value

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
            try:
                self._save(self.filename)
            except Exception as e:
                print(f'Exception: {e.__name__}: {e}')

        # Update output params with current values
        params_str = '\n'.join(f'{k}: {v}' for k, v in
        {
            'dt': self.t_to_seconds(self.dt),
            'thr1': self.thr1,
            'thr2': self.thr2,
            'status': self.status_string,
            'output_tag': self.filename,
        }.items())
        self.outputs['out_params'].set_value(params_str)
        self.outputs['out_params'].generation_time = self.current_time




