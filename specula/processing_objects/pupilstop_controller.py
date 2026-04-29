from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.lib.extrapolation_2d import EFInterpolator


class PupilstopController(BaseProcessingObj):
    """
    Processing object that updates a Pupilstop object over time.

    The object always triggers at each iteration so that the output
    pupilstop generation_time is refreshed regularly, even with static inputs.

    Optional BaseValue inputs can drive geometry updates:
      - in_rotation_deg: scalar rotation angle [deg]
      - in_shift_xy_px: 2-element shift [x, y] in pixels
      - in_magnification: scalar magnification factor (>0)

    If any of the optional inputs are connected, the amplitude mask is regenerated
    every trigger from the initial mask applying the current geometry.
    """

    def __init__(self,
                 pupilstop: Pupilstop,
                 threshold_mask: bool = True,
                 mask_threshold: float = 0.5,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters
        ----------
        
        pupilstop: Pupilstop
            The Pupilstop object to be controlled and updated.
        threshold_mask: bool, optional
            If True, the updated mask will be thresholded to binary values based on mask_threshold 
            (default: True).
        mask_threshold: float, optional
            Threshold value for binarizing the mask if threshold_mask is True (default: 0.5).
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None
            (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).  
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        in_shift_xy = cpuArray(pupilstop.shiftXYinPixel).astype(float)
        self._pupilstop = pupilstop

        # Keep an internal source layer on this object's device and never mutate input pupilstop.
        self._in_layer = Layer(
            dimx=pupilstop.size[1],
            dimy=pupilstop.size[0],
            pixel_pitch=pupilstop.pixel_pitch,
            height=0,
            shiftXYinPixel=in_shift_xy,
            rotInDeg=float(pupilstop.rotInDeg),
            magnification=float(pupilstop.magnification),
            target_device_idx=self.target_device_idx,
            precision=self.precision,
        )
        self._in_layer.A[:] = self.to_xp(pupilstop.A)
        # phase is zero for pupilstop

        self._out_layer = Layer(
            dimx=pupilstop.size[1],
            dimy=pupilstop.size[0],
            pixel_pitch=pupilstop.pixel_pitch,
            height=0,
            shiftXYinPixel=in_shift_xy,
            rotInDeg=float(pupilstop.rotInDeg),
            magnification=float(pupilstop.magnification),
            target_device_idx=self.target_device_idx,
            precision=self.precision,
        )
        self._out_layer.A[:] = self.to_xp(pupilstop.A)
        # phase is zero for pupilstop

        self.outputs['out_layer'] = self._out_layer

        self.inputs['in_rotation_deg'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_shift_xy_px'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_magnification'] = InputValue(type=BaseValue, optional=True)

        self.update_mask = False
        self.threshold_mask = threshold_mask
        self.mask_threshold = mask_threshold

    @classmethod
    def input_names(cls):
        return {
            'in_rotation_deg': InputDesc(BaseValue, 'Scalar rotation angle [deg] (optional)'),
            'in_shift_xy_px': InputDesc(BaseValue, '[x, y] shift in pixels (optional)'),
            'in_magnification': InputDesc(BaseValue, 'Scalar magnification factor (>0) (optional)'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_layer': OutputDesc(Layer, 'Updated pupilstop layer'),
        }


    def setup(self):
        super().setup()
        self.update_mask = any(
            self.local_inputs[k] is not None
            for k in ('in_rotation_deg', 'in_shift_xy_px', 'in_magnification')
        )
        if self.update_mask:
            self._ef_interpolator = EFInterpolator(
                in_ef=self._in_layer,
                out_shape=self._in_layer.size,
                rotAnglePhInDeg=float(self._out_layer.rotInDeg),
                xShiftPhInPixel=float(self._out_layer.shiftXYinPixel[0]),
                yShiftPhInPixel=float(self._out_layer.shiftXYinPixel[1]),
                magnification=float(self._out_layer.magnification),
                mask_threshold=self.mask_threshold,
                force_extrapolation=True,
                use_out_ef_cache=False,
                target_device_idx=self.target_device_idx,
                precision=self.precision,
            )

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        in_rot = self.local_inputs['in_rotation_deg']
        if in_rot is not None and in_rot.value is not None:
            arr = cpuArray(in_rot.value).ravel()
            if arr.size != 1:
                raise ValueError(f"in_rotation_deg must be scalar, got size={arr.size}")
            self._out_layer.rotInDeg = float(arr[0])

        in_shift = self.local_inputs['in_shift_xy_px']
        if in_shift is not None and in_shift.value is not None:
            arr = cpuArray(in_shift.value).ravel()
            if arr.size != 2:
                raise ValueError(f"in_shift_xy_px must contain exactly 2 values [x, y],"
                                 f" got size={arr.size}")
            self._out_layer.shiftXYinPixel = arr.astype(float)

        in_mag = self.local_inputs['in_magnification']
        if in_mag is not None and in_mag.value is not None:
            arr = cpuArray(in_mag.value).ravel()
            if arr.size != 1:
                raise ValueError(f"in_magnification must be scalar, got size={arr.size}")
            value = float(arr[0])
            if value <= 0:
                raise ValueError(f"in_magnification must be > 0, got {value}")
            self._out_layer.magnification = value


    def trigger_code(self):

        if self.update_mask:
            self._ef_interpolator.update_parameters(
                xShiftPhInPixel=float(self._out_layer.shiftXYinPixel[0]),
                yShiftPhInPixel=float(self._out_layer.shiftXYinPixel[1]),
                rotAnglePhInDeg=float(self._out_layer.rotInDeg),
                magnification=float(self._out_layer.magnification),
            )
            self._ef_interpolator.interpolate()
            mask = self._ef_interpolator.interpolated_ef().A

            if self.threshold_mask:
                mask = mask >= self.mask_threshold

            self._out_layer.A[:] = mask


    def post_trigger(self):
        super().post_trigger()
        self._out_layer.generation_time = self.current_time
