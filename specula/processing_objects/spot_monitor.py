import numpy as _np
from astropy.modeling import models as _models, fitting as _fitting

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.lib.utils import unravel_index_2d
from specula import cpuArray


class SpotMonitor(BaseProcessingObj):
    """
    SpotMonitor
    
    Monitors wavefront sensor spot quality by fitting a 2D Moffat profile.
    
    - Input: Pixels (full WFS image)
    - Uses SubapData to extract valid sub-apertures
    - Sums all sub-apertures into a single np_sub x np_sub image
    - Fits the summed spot with a 2D Moffat + constant sky (using astropy)
    - Outputs: fit parameters, model image, residuals, summed image
    
    Parameters
    ----------
    subapdata : SubapData
        Subaperture geometry and indexing information
    initial_alpha : float, optional
        Initial guess for Moffat alpha parameter (default: 2.0)
    initial_gamma : float, optional
        Initial guess for Moffat gamma parameter (default: 3.0)
    bounds_alpha : tuple, optional
        Bounds for alpha parameter (default: (0.5, 10.0))
    bounds_gamma : tuple, optional
        Bounds for gamma parameter (default: (0.1, None))
    target_device_idx : int, optional
        Target device index
    precision : int, optional
        Numerical precision
        
    Attributes
    ----------
    outputs['out_sum_pixels'] : Pixels
        Summed subaperture image (np_sub x np_sub)
    outputs['out_model_pixels'] : Pixels
        Fitted Moffat model image
    outputs['out_residual_pixels'] : Pixels
        Residuals (data - model)
    outputs['out_params'] : BaseValue
        Array with 9 parameters:
        [0] amplitude - Moffat amplitude
        [1] x0 - centroid x position
        [2] y0 - centroid y position
        [3] gamma - Moffat gamma parameter
        [4] alpha - Moffat alpha parameter
        [5] sky - constant background level
        [6] fwhm - Full Width Half Maximum
        [7] chi2 - mean squared residual
        [8] success - 1.0 if fit converged, 0.0 otherwise
    """

    def __init__(self,
                 subapdata: SubapData,
                 initial_alpha: float = 2.0,
                 initial_gamma: float = 3.0,
                 bounds_alpha=(0.5, 10.0),
                 bounds_gamma=(0.1, None),
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if subapdata is None:
            raise ValueError("SpotMonitor requires a valid SubapData instance")
        self.subapdata = subapdata

        # Inputs
        self.inputs['in_pixels'] = InputValue(type=Pixels)

        # Internal holders
        self._initial_alpha = float(initial_alpha)
        self._initial_gamma = float(initial_gamma)
        self._bounds_alpha = bounds_alpha
        self._bounds_gamma = bounds_gamma

        # Output artifacts sizes (np_sub x np_sub)
        np_sub = self.subapdata.np_sub

        # Outputs
        self.sum_pixels = Pixels(np_sub, np_sub, target_device_idx=self.target_device_idx)
        self.model_pixels = Pixels(np_sub, np_sub, target_device_idx=self.target_device_idx)
        self.residual_pixels = Pixels(np_sub, np_sub, target_device_idx=self.target_device_idx)

        # parameters: [amplitude, x0, y0, gamma, alpha, sky, fwhm, chi2, success]
        self.params = BaseValue(value=self.xp.zeros(9, dtype=self.dtype),
                                target_device_idx=self.target_device_idx)

        self.outputs['out_sum_pixels'] = self.sum_pixels
        self.outputs['out_model_pixels'] = self.model_pixels
        self.outputs['out_residual_pixels'] = self.residual_pixels
        self.outputs['out_params'] = self.params

    def _estimate_initials(self, img):
        """Estimate initial parameters from image.
        
        Parameters
        ----------
        img : array
            2D image array (np_sub x np_sub)
            
        Returns
        -------
        tuple
            (amplitude, x0, y0, gamma, alpha, sky)
        """
        np_sub = self.subapdata.np_sub
        # Sky from border pixels
        border = self.xp.hstack([
            img[0, :], img[-1, :],
            img[1:-1, 0], img[1:-1, -1]
        ])
        sky0 = float(self.xp.median(border))
        amp0 = float(self.xp.clip(self.xp.max(img) - sky0, 1e-6, None))
        cx0 = cy0 = (np_sub - 1) / 2.0
        alpha0 = float(self._initial_alpha)
        gamma0 = float(self._initial_gamma)
        return amp0, cx0, cy0, gamma0, alpha0, sky0

    @staticmethod
    def _moffat_fwhm(gamma, alpha):
        """Calculate FWHM for Moffat profile.
        
        Parameters
        ----------
        gamma : float
            Moffat gamma parameter
        alpha : float
            Moffat alpha parameter
            
        Returns
        -------
        float
            Full Width Half Maximum
        """
        # FWHM for Moffat2D: 2 * gamma * sqrt(2^(1/alpha) - 1)
        import numpy as np
        return float(2.0 * gamma * np.sqrt(2.0 ** (1.0 / alpha) - 1.0))

    def trigger_code(self):
        """Main processing: sum subapertures and fit Moffat profile."""
        # Get input pixels
        in_pixels = self.local_inputs['in_pixels'].pixels  # xp array
        np_sub = self.subapdata.np_sub

        # Extract subaperture pixels and sum them
        idx2d = unravel_index_2d(self.subapdata.idxs, in_pixels.shape, self.xp)
        # shape: (np_sub*np_sub, n_subaps)
        sub_stack = in_pixels[idx2d].T.astype(self.dtype, copy=False)
        sum_flat = self.xp.sum(sub_stack, axis=1)  # length np_sub*np_sub
        sum_img = sum_flat.reshape((np_sub, np_sub))
        self.sum_pixels.pixels[:] = sum_img

        # Fit Moffat + constant sky on CPU (astropy needs numpy arrays)
        img_cpu = cpuArray(sum_img).astype('float64', copy=False)
        yy, xx = _np.mgrid[0:np_sub, 0:np_sub]  # note: (y, x) indexing

        # Initial guesses
        amp0, x0, y0, gamma0, alpha0, sky0 = self._estimate_initials(sum_img)

        # Compose model: Const2D + Moffat2D
        const = _models.Const2D(amplitude=sky0, bounds={'amplitude': (0.0, None)})
        moff = _models.Moffat2D(amplitude=amp0,
                                x_0=x0, y_0=y0,
                                gamma=gamma0, alpha=alpha0,
                                bounds={'amplitude': (0.0, None),
                                        'gamma': (self._bounds_gamma[0], self._bounds_gamma[1]),
                                        'alpha': (self._bounds_alpha[0], self._bounds_alpha[1])})
        model = const + moff

        fitter = _fitting.LevMarLSQFitter()

        success = 0.0
        chi2 = _np.nan
        try:
            fit = fitter(model, xx, yy, img_cpu)
            model_img = fit(xx, yy)

            # Residuals and metrics
            res_img = img_cpu - model_img
            chi2 = float(_np.mean(res_img**2))
            success = 1.0

            # Save model/residual back to device
            self.model_pixels.pixels[:] = self.to_xp(model_img.astype(img_cpu.dtype))
            self.residual_pixels.pixels[:] = self.to_xp(res_img.astype(img_cpu.dtype))

            # Extract fitted parameters
            # model = Const2D + Moffat2D -> parameters are additive, access via submodels
            sky = float(fit[0].amplitude.value)
            amp = float(fit[1].amplitude.value)
            x0f = float(fit[1].x_0.value)
            y0f = float(fit[1].y_0.value)
            gammaf = float(fit[1].gamma.value)
            alphaf = float(fit[1].alpha.value)
            fwhm = self._moffat_fwhm(gammaf, alphaf)

        except Exception:  # pragma: no cover
            # On failure, zero model/residual
            self.model_pixels.pixels[:] = 0
            self.residual_pixels.pixels[:] = 0
            sky = sky0
            amp = amp0
            x0f, y0f = x0, y0
            gammaf, alphaf = gamma0, alpha0
            fwhm = self._moffat_fwhm(gammaf, alphaf)

        # Save params on device
        out = self.params.value
        out[0] = amp
        out[1] = x0f
        out[2] = y0f
        out[3] = gammaf
        out[4] = alphaf
        out[5] = sky
        out[6] = fwhm
        out[7] = chi2
        out[8] = success

    def post_trigger(self):
        """Update generation times for all outputs."""
        super().post_trigger()
        # Propagate generation time
        self.outputs['out_sum_pixels'].generation_time = self.current_time
        self.outputs['out_model_pixels'].generation_time = self.current_time
        self.outputs['out_residual_pixels'].generation_time = self.current_time
        self.outputs['out_params'].generation_time = self.current_time

    @property
    def amplitude(self):
        """Get fitted Moffat amplitude."""
        return float(cpuArray(self.params.value[0]))

    @property
    def centroid(self):
        """Get fitted centroid position (x0, y0)."""
        return tuple(cpuArray(self.params.value[1:3]))

    @property
    def fwhm(self):
        """Get fitted FWHM."""
        return float(cpuArray(self.params.value[6]))

    @property
    def sky_level(self):
        """Get fitted sky background level."""
        return float(cpuArray(self.params.value[5]))

    @property
    def fit_quality(self):
        """Get fit quality metrics (chi2, success)."""
        vals = cpuArray(self.params.value[7:9])
        return {'chi2': float(vals[0]), 'success': bool(vals[1])}