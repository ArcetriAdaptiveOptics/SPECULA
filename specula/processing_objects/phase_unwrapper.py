from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula import np


from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula import np


class PhaseUnwrapperBasic(BaseProcessingObj):
    """
    Phase unwrapper using synthetic wavelength method.
    
    Unwraps piston values from two measurements at different wavelengths using a 
    synthetic wavelength derived from the two input wavelengths. Employs integer 
    search over possible phase wraps to find the best matching unwrapped phase.
    
    The algorithm computes a synthetic wavelength and conducts an exhaustive search 
    over integer multiples of the first wavelength to identify the phase value that 
    best minimizes the residual when compared against the second wavelength measurement.
    """

    def __init__(self,
                 lambda_1: float,
                 lambda_2: float,
                 max_capture: float = None,
                 target_device_idx=None,
                 precision=None):
        """
        Initialize the WLI phase unwrapper.

        Parameters
        ----------
        lambda_1 : float
            First measurement wavelength [microns].
        lambda_2 : float
            Second measurement wavelength [microns]. Must be different from lambda_1.
        max_capture : float, optional
            Maximum capture range in microns. If None, defaults to synthetic_lambda/2.
            Determines search range for phase unwrapping.
        target_device_idx : int, optional
            Device index (-1=CPU, >=0=GPU). Defaults to global default.
        precision : int, optional
            Precision flag (0=double, 1=single). Defaults to global default.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Validate wavelengths
        if lambda_1 <= 0 or lambda_2 <= 0:
            raise ValueError("Wavelengths must be positive")
        if abs(lambda_1 - lambda_2) < 1e-10:
            raise ValueError("Wavelengths must be different")

        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)

        # Compute synthetic wavelength (beat wavelength)
        self.synthetic_lambda = (
            self.lambda_1 * self.lambda_2 / abs(self.lambda_1 - self.lambda_2)
        )

        # Maximum capture range (determines unambiguous phase range)
        if max_capture is None:
            self.max_capture = self.synthetic_lambda / 2.0
        else:
            self.max_capture = float(max_capture)

        # Integer search range: how many wavelengths fit in max capture range
        self.max_k = int(
            self.xp.ceil(self.max_capture / self.lambda_1)
        ) + 1

        # Pre-allocate output array (shape determined in setup)
        self.out_pistons_array = None
        self.nseg_cached = None

        # Define inputs and outputs
        self.inputs['in_pistons_1'] = InputValue(type=BaseValue, optional=False)
        self.inputs['in_pistons_2'] = InputValue(type=BaseValue, optional=False)

        # Output data object
        self.out_pistons_data = BaseValue(
            description='Unwrapped piston values from WLI phase unwrapping',
            target_device_idx=target_device_idx
        )
        self.outputs['out_pistons'] = self.out_pistons_data

    def setup(self):
        """
        Prepare for simulation by allocating output arrays based on input shapes.
        """
        super().setup()

        # Get input shapes to pre-allocate output
        p1 = self.local_inputs['in_pistons_1']
        p2 = self.local_inputs['in_pistons_2']

        if p1 is None:
            raise ValueError("in_pistons_1 must be provided")
        if p2 is None:
            raise ValueError("in_pistons_2 must be provided")

        # Get array representation and shape
        p1_array = self.xp.asarray(p1.get_value() if hasattr(p1, 'get_value') else p1)
        p2_array = self.xp.asarray(p2.get_value() if hasattr(p2, 'get_value') else p2)

        if p1_array.shape != p2_array.shape:
            raise ValueError(
                f"Input shapes must match: in_pistons_1={p1_array.shape} "
                f"vs in_pistons_2={p2_array.shape}"
            )

        self.nseg_cached = p1_array.shape[0]

        # Pre-allocate output array (never reallocated, just reused via [:] assignment)
        self.out_pistons_array = self.xp.zeros(self.nseg_cached, dtype=self.dtype)
        self.out_pistons_data.value = self.out_pistons_array

    def prepare_trigger(self, t):
        """
        Ensure correct device is active before trigger_code() execution.
        """
        super().prepare_trigger(t)

    @staticmethod
    def wrap_phase(x, wavelength):
        """
        Wrap phase values into [-wavelength/2, wavelength/2) interval.

        Implements modular arithmetic to keep phase differences bounded within
        a single wavelength period, essential for phase comparison.

        Parameters
        ----------
        x : array-like
            Phase values to wrap [microns].
        wavelength : float
            Reference wavelength [microns].

        Returns
        -------
        wrapped : ndarray
            Phase values wrapped into [-wavelength/2, wavelength/2).
        """
        # ((x + λ/2) mod λ) - λ/2 maps to [-λ/2, λ/2)
        return ((x + wavelength / 2.0) % wavelength) - wavelength / 2.0


    def trigger_code(self):
        """
        Execute phase unwrapping algorithm for all segments.

        For each piston measurement:
        1. Generate candidate unwrapped values by adding integer multiples of lambda_1
        2. For each candidate, compute residual = |candidate(mod lambda_2) - p2(mod lambda_2)|
        3. Select candidate with minimum residual within [-max_capture, +max_capture]
        """
        p1 = self.local_inputs['in_pistons_1'].get_value()
        p2 = self.local_inputs['in_pistons_2'].get_value()

        p1 = self.xp.asarray(p1, dtype=self.dtype)
        p2 = self.xp.asarray(p2, dtype=self.dtype)

        nseg = p1.shape[0]

        # Generate integer search space
        k_values = self.xp.arange(
            -self.max_k,
            self.max_k + 1,
            dtype=self.xp.int32,
        )

        # Vectorized processing per segment
        for i in range(nseg):
            # Candidate unwrapped phase values: p1[i] + k * lambda_1
            candidates = p1[i] + k_values * self.lambda_1

            # Wrap candidates to lambda_2 domain for comparison
            candidates_mod_lambda2 = self.wrap_phase(candidates, self.lambda_2)
            p2_mod_lambda2 = self.wrap_phase(p2[i], self.lambda_2)

            # Residual: how well does each candidate match p2 measurement?
            residuals = self.xp.abs(candidates_mod_lambda2 - p2_mod_lambda2)

            # Filter candidates within valid capture range
            valid_mask = self.xp.abs(candidates) <= self.max_capture
            
            if self.xp.any(valid_mask):
                filtered_residuals = self.xp.where(
                    valid_mask,
                    residuals,
                    self.xp.inf
                )
                best_idx = self.xp.argmin(filtered_residuals)
            else:
                best_idx = self.xp.argmin(residuals)

            # Return the unwrapped candidate value
            self.out_pistons_array[i] = candidates[best_idx]
    
    def post_trigger(self):
        """
        Set generation time and synchronize if using CUDA graphs.
        """
        super().post_trigger()
        self.out_pistons_data.generation_time = self.current_time

    @classmethod
    def input_names(cls):
        """Declare expected input names and types."""
        return {
            'in_pistons_1': (BaseValue, 'Piston values at wavelength lambda_1'),
            'in_pistons_2': (BaseValue, 'Piston values at wavelength lambda_2'),
        }

    @classmethod
    def output_names(cls):
        """Declare expected output names and types."""
        return {
            'out_pistons': (BaseValue, 'Unwrapped piston values'),
        }

        
class PhaseUnwrapper(BaseProcessingObj):
    """
    Phase unwrapper.
    """

    def __init__(self,
                 lambda_1: float,
                 lambda_2: float,
                 max_capture: float = None,
                 confidence_threshold: float = 0.2,
                 outlier_residual_threshold: float = 200,
                 use_median_filter: bool = True,
                 target_device_idx=None,
                 precision=None):
        """
        Parameters
        ----------
        confidence_threshold : float
            Minimum normalized confidence score (0-1). Solutions below this are flagged.
        outlier_residual_threshold : float
            Max acceptable residual. If None, defaults to lambda_2/4.
        use_median_filter : bool
            Apply spatial median filtering to detect local anomalies.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Validate wavelengths
        if lambda_1 <= 0 or lambda_2 <= 0:
            raise ValueError("Wavelengths must be positive")
        if abs(lambda_1 - lambda_2) < 1e-10:
            raise ValueError("Wavelengths must be different")

        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.confidence_threshold = float(confidence_threshold)
        self.use_median_filter = use_median_filter

        # Compute synthetic wavelength
        self.synthetic_lambda = (
            self.lambda_1 * self.lambda_2 / abs(self.lambda_1 - self.lambda_2)
        )

        # Maximum capture range
        if max_capture is None:
            self.max_capture = self.synthetic_lambda / 2.0
        else:
            self.max_capture = float(max_capture)

        # Outlier threshold
        if outlier_residual_threshold is None:
            self.outlier_residual_threshold = self.lambda_2 / 4.0
        else:
            self.outlier_residual_threshold = float(outlier_residual_threshold)

        # Integer search range
        self.max_k = int(
            self.xp.ceil(self.max_capture / self.lambda_1)
        ) + 1

        # Pre-allocate output arrays
        self.out_pistons_array = None
        self.confidences = None
        self.outlier_flags = None
        self.residuals = None

        # Define inputs and outputs
        self.inputs['in_pistons_1'] = InputValue(type=BaseValue, optional=False)
        self.inputs['in_pistons_2'] = InputValue(type=BaseValue, optional=False)

        self.out_pistons_data = BaseValue(
            description='Unwrapped piston values from WLI phase unwrapping',
            target_device_idx=target_device_idx
        )
        self.outputs['out_pistons'] = self.out_pistons_data

    def setup(self):
        """Prepare for simulation by allocating output arrays."""
        super().setup()

        p1 = self.local_inputs['in_pistons_1']
        p2 = self.local_inputs['in_pistons_2']

        if p1 is None or p2 is None:
            raise ValueError("Both in_pistons_1 and in_pistons_2 must be provided")

        p1_array = self.xp.asarray(p1.get_value() if hasattr(p1, 'get_value') else p1)
        p2_array = self.xp.asarray(p2.get_value() if hasattr(p2, 'get_value') else p2)

        if p1_array.shape != p2_array.shape:
            raise ValueError(
                f"Input shapes must match: in_pistons_1={p1_array.shape} "
                f"vs in_pistons_2={p2_array.shape}"
            )

        nseg = p1_array.shape[0]

        # Pre-allocate arrays
        self.out_pistons_array = self.xp.zeros(nseg, dtype=self.dtype)
        self.confidences = self.xp.zeros(nseg, dtype=self.dtype)
        self.outlier_flags = self.xp.zeros(nseg, dtype=bool)
        self.residuals = self.xp.zeros(nseg, dtype=self.dtype)

        self.out_pistons_data.value = self.out_pistons_array

    @staticmethod
    def wrap_phase(x, wavelength):
        """Wrap phase values into [-wavelength/2, wavelength/2) interval."""
        return ((x + wavelength / 2.0) % wavelength) - wavelength / 2.0

    def _apply_spatial_filtering(self, estimates, nseg, kernel_size=3):
        """
        Apply spatial median filter to smooth estimates and identify outliers.
        Compares raw solution to median-filtered version.
        """
        if nseg < kernel_size or not self.use_median_filter:
            return estimates

        # Compute local median (simple sliding window)
        from scipy.ndimage import median_filter as scipy_median
        estimates_cpu = self.xp.asnumpy(estimates) if hasattr(self.xp, 'asnumpy') else estimates
        
        median_filtered = scipy_median(estimates_cpu, size=kernel_size)
        median_filtered = self.xp.asarray(median_filtered, dtype=self.dtype)

        return median_filtered

    def trigger_code(self):
        """
        Execute robust phase unwrapping with confidence assessment.
        """
        p1 = self.local_inputs['in_pistons_1'].get_value()
        p2 = self.local_inputs['in_pistons_2'].get_value()

        p1 = self.xp.asarray(p1, dtype=self.dtype)
        p2 = self.xp.asarray(p2, dtype=self.dtype)

        nseg = p1.shape[0]

        # Generate integer search space
        k_values = self.xp.arange(
            -self.max_k,
            self.max_k + 1,
            dtype=self.xp.int32,
        )

        # Process each segment
        for i in range(nseg):
            candidates = p1[i] + k_values * self.lambda_1
            wrapped_candidates = self.wrap_phase(candidates, self.lambda_2)
            
            residuals = self.xp.abs(
                self.wrap_phase(
                    wrapped_candidates - p2[i],
                    self.lambda_2,
                )
            )

            # Filter by capture range
            valid_mask = self.xp.abs(candidates) <= self.max_capture
            filtered_residuals = self.xp.where(
                valid_mask,
                residuals,
                self.xp.inf
            )

            # Find best and second-best solutions
            best_idx = self.xp.argmin(filtered_residuals)
            best_residual = filtered_residuals[best_idx]
            best_candidate = candidates[best_idx]

            # **Robustness: Confidence score**
            sorted_residuals = self.xp.sort(filtered_residuals)
            if len(sorted_residuals) > 1 and self.xp.isfinite(sorted_residuals[1]):
                residual_gap = sorted_residuals[1] - sorted_residuals[0]
                confidence = min(1.0, residual_gap / (self.lambda_2 / 2.0))
            else:
                confidence = 0.0

            # **Robustness: Outlier detection**
            is_outlier = best_residual > self.outlier_residual_threshold
            is_low_confidence = confidence < self.confidence_threshold

            # **Robustness: Multi-wavelength consistency check**
            # Verify that the best candidate wraps consistently on both wavelengths
            wrap1 = self.wrap_phase(best_candidate, self.lambda_1)
            wrap2 = self.wrap_phase(best_candidate, self.lambda_2)
            consistency = self.xp.abs(
                self.wrap_phase(wrap1 - wrap2, self.lambda_2)
            )
            is_inconsistent = consistency > (self.lambda_2 / 10.0)  # Allow 10% tolerance

            # Store metrics
            self.confidences[i] = confidence
            self.outlier_flags[i] = is_outlier or is_low_confidence or is_inconsistent
            self.residuals[i] = best_residual

            self.out_pistons_array[i] = best_candidate

        # **Optional: Spatial filtering to smooth field**
        if self.use_median_filter and nseg > 1:
            smoothed = self._apply_spatial_filtering(
                self.out_pistons_array, nseg
            )
            # Only update outliers with smoothed values
            self.out_pistons_array[:] = self.xp.where(
                self.outlier_flags,
                smoothed,
                self.out_pistons_array
            )

    def post_trigger(self):
        """Set generation time and synchronize if using CUDA graphs."""
        super().post_trigger()
        self.out_pistons_data.generation_time = self.current_time

    @classmethod
    def input_names(cls):
        return {
            'in_pistons_1': (BaseValue, 'Piston values at wavelength lambda_1'),
            'in_pistons_2': (BaseValue, 'Piston values at wavelength lambda_2'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_pistons': (BaseValue, 'Unwrapped piston values'),
        }
    
