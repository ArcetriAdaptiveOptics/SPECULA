from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.electric_field import ElectricField
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

        if pupilstop.target_device_idx != self.target_device_idx:
            raise ValueError(
                f"PupilstopController and input pupilstop must use the same target_device_idx "
                f"({self.target_device_idx} != {pupilstop.target_device_idx})"
            )

        self._pupilstop = pupilstop
        self.outputs['out_layer'] = self._pupilstop

        self.inputs['in_rotation_deg'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_shift_xy_px'] = InputValue(type=BaseValue, optional=True)
        self.inputs['in_magnification'] = InputValue(type=BaseValue, optional=True)

        self.update_mask = False
        self.threshold_mask = threshold_mask
        self.mask_threshold = mask_threshold

        # Normalise shiftXYinPixel to a float numpy array (may be a tuple at construction).
        self._pupilstop.shiftXYinPixel = cpuArray(self._pupilstop.shiftXYinPixel).astype(float)

        # Keep an immutable reference mask to avoid cumulative interpolation artifacts.
        self._base_mask = self.to_xp(self._pupilstop.A, dtype=self._pupilstop.dtype, force_copy=True)

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
            self._base_ef = ElectricField(
                dimx=self._pupilstop.size[1],
                dimy=self._pupilstop.size[0],
                pixel_pitch=self._pupilstop.pixel_pitch,
                target_device_idx=self.target_device_idx,
                precision=self.precision,
            )
            self._base_ef.A[:] = self._base_mask
            self._base_ef.phaseInNm[:] = 0

            self._ef_interpolator = EFInterpolator(
                in_ef=self._base_ef,
                out_shape=self._base_ef.size,
                rotAnglePhInDeg=float(self._pupilstop.rotInDeg),
                xShiftPhInPixel=float(self._pupilstop.shiftXYinPixel[0]),
                yShiftPhInPixel=float(self._pupilstop.shiftXYinPixel[1]),
                magnification=float(self._pupilstop.magnification),
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
            self._pupilstop.rotInDeg = float(arr[0])

        in_shift = self.local_inputs['in_shift_xy_px']
        if in_shift is not None and in_shift.value is not None:
            arr = cpuArray(in_shift.value).ravel()
            if arr.size != 2:
                raise ValueError(f"in_shift_xy_px must contain exactly 2 values [x, y],"
                                 f" got size={arr.size}")
            self._pupilstop.shiftXYinPixel = arr.astype(float)

        in_mag = self.local_inputs['in_magnification']
        if in_mag is not None and in_mag.value is not None:
            arr = cpuArray(in_mag.value).ravel()
            if arr.size != 1:
                raise ValueError(f"in_magnification must be scalar, got size={arr.size}")
            value = float(arr[0])
            if value <= 0:
                raise ValueError(f"in_magnification must be > 0, got {value}")
            self._pupilstop.magnification = value


    def trigger_code(self):

        if self.update_mask:
            self._ef_interpolator.update_parameters(
                xShiftPhInPixel=float(self._pupilstop.shiftXYinPixel[0]),
                yShiftPhInPixel=float(self._pupilstop.shiftXYinPixel[1]),
                rotAnglePhInDeg=float(self._pupilstop.rotInDeg),
                magnification=float(self._pupilstop.magnification),
            )
            self._ef_interpolator.interpolate()
            mask = self._ef_interpolator.interpolated_ef().A

            mask = self.xp.asarray(mask, dtype=self._pupilstop.dtype)
            if self.threshold_mask:
                mask = (mask >= self.mask_threshold).astype(self._pupilstop.dtype)

            self._pupilstop.A[:] = mask


    def post_trigger(self):
        super().post_trigger()
        self._pupilstop.generation_time = self.current_time
