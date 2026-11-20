from specula.base_processing_obj import InputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.lib.make_xy import make_xy

from specula.processing_objects.sh_slopec import ShSlopec


class ShSlopecMovable(ShSlopec):
    """
    ShSlopec whose WCoG window and slope zero move together.

    An optional input ``in_window`` carries ``[wx, wy, hold]``: the window centre in
    detector pixels relative to the subaperture centre (x = column, y = row) and a hold
    flag. The Gaussian weight mask is centred on ``(wx, wy)`` and ``(wx, wy)`` is subtracted
    from the slopes, so the output is the centroid relative to the window: the loop keeps
    the calibrated gain and drives the spot to the window centre. With ``hold`` set the
    slopes are zero (the control loop stops integrating). Unconnected or ``w = 0``: identical
    to :class:`ShSlopec`.

    Only the Gaussian WCoG (``windowing=False``, ``exp_weight=1``, no quadcell) is supported.
    """

    def __init__(self,
                 subapdata: SubapData,
                 sn: Slopes = None,
                 thr_value: float = -1,
                 exp_weight: float = 1.0,
                 filtmat=None,
                 weightedPixRad: float = 0.0,
                 windowing: bool = False,
                 weight_int_pixel_dt: float = 0,
                 window_int_pixel: bool = False,
                 window_int_threshold: float = 1.0,
                 vecWeiPixRadT: list = None,
                 interleave: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        # explicit signature (same as ShSlopec): the simulation builder reads the type hints
        super().__init__(subapdata=subapdata, sn=sn, thr_value=thr_value, exp_weight=exp_weight,
                         filtmat=filtmat, weightedPixRad=weightedPixRad, windowing=windowing,
                         weight_int_pixel_dt=weight_int_pixel_dt, window_int_pixel=window_int_pixel,
                         window_int_threshold=window_int_threshold, vecWeiPixRadT=vecWeiPixRadT,
                         interleave=interleave, target_device_idx=target_device_idx, precision=precision)
        if self.windowing or self.quadcell_mode or self.exp_weight != 1.0:
            raise ValueError('ShSlopecMovable supports only the Gaussian WCoG '
                             '(windowing=False, exp_weight=1, quadcell off)')
        if self.weighted_pix_rad == 0:
            raise ValueError('ShSlopecMovable needs weightedPixRad > 0')
        self._w = self.xp.zeros(2, dtype=self.dtype)
        self._w_applied = (0.0, 0.0)
        self._hold = False
        self.inputs['in_window'] = InputValue(type=BaseValue, optional=True)

    @classmethod
    def input_names(cls):
        result = super().input_names()
        result['in_window'] = InputDesc(BaseValue, 'Window centre [wx, wy] in px and hold flag [0/1]')
        return result

    def set_xy_weights(self):
        super().set_xy_weights()
        w = getattr(self, '_w_applied', (0.0, 0.0))
        if w != (0.0, 0.0):
            self._shift_weights(*w)

    def _shift_weights(self, wx, wy):
        """Recompute mask and coordinate weights for a window centred on (wx, wy) px."""
        import numpy as np
        if self.weighted_pix_rad == 0:                 # radius 0 from vecWeiPixRadT: no weighting, keep the base weights
            return
        n = self.subapdata.np_sub
        x, y = make_xy(n, 1.0, xp=np, dtype=self.dtype)
        c = (n - 1) / 2.0
        sigma = 2.0 * self.weighted_pix_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        cols = np.exp(-0.5 * ((np.arange(n) - c - wx) / sigma) ** 2)
        rows = np.exp(-0.5 * ((np.arange(n) - c - wy) / sigma) ** 2)
        mask = np.outer(rows, cols)                   # axis 0 = y (rows), axis 1 = x (columns)
        mask /= np.max(mask)
        mask[mask < 1e-6] = 0.0
        self.mask_weighted = self.to_xp(mask.astype(self.dtype))
        self.xweights = self.to_xp((x * mask).astype(self.dtype))
        self.yweights = self.to_xp((y * mask).astype(self.dtype))
        self.xweights_flat = self.xweights.reshape(n * n, 1)
        self.yweights_flat = self.yweights.reshape(n * n, 1)
        self.mask_weighted_flat = self.mask_weighted.reshape(n * n, 1)
        # the slopes are computed from the combined (denominator, x, y) array: refresh it, in place
        self._weights[:] = self.xp.vstack([self.mask_weighted.ravel(), self.xweights.ravel(), self.yweights.ravel()])

    def compute_slopes(self):
        win = self.local_inputs.get('in_window')
        if win is not None:
            v = self.to_xp(win.value)
            wx, wy = float(v[0]), float(v[1])
            self._hold = bool(v[2] > 0.5) if v.size > 2 else False
        else:
            wx = wy = 0.0
            self._hold = False
        if (wx, wy) != self._w_applied:
            self._w_applied = (wx, wy)
            self.set_xy_weights()
        self._w[:] = self.xp.asarray([wx, wy], dtype=self.dtype)
        super().compute_slopes()
        # slopes are in units of half a subaperture: 1 px = 2 / np_sub
        px = 2.0 / self.subapdata.np_sub
        if self._hold:
            # zero after apply_slopes_corrections() has subtracted the slope null (if any)
            sn = self.sn
            self.slopes.xslopes = (sn.xslopes.copy() if sn else self.xp.zeros_like(self.slopes.xslopes))
            self.slopes.yslopes = (sn.yslopes.copy() if sn else self.xp.zeros_like(self.slopes.yslopes))
        else:
            self.slopes.xslopes = self.slopes.xslopes - self._w[0] * px
            self.slopes.yslopes = self.slopes.yslopes - self._w[1] * px
