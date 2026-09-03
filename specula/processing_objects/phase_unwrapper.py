from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula import np

    
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula import np

class PhaseUnwrapper(BaseProcessingObj):
    """
    Robust multi-wavelength interferometry (WLI) phase unwrapper with modular behavior control.
    
    Optimized for closed-loop control systems with:
    - Fast edge response (minimal latency)
    - Optional temporal filtering (with proper blending, not replacement)
    - Corrected Stage 1/Stage 2 decision logic
    
    Behaviors controlled via boolean flags for easy debugging:
    - unwrap_enabled: If False, output = input (no processing)
    - two_stage_enabled: Use two-stage or always full unwrapping
    - temporal_filtering_mode: 'none', 'median', or 'weighted_average'
    
    Two-stage logic (when enabled):
    - Stage 1: Use p1 directly as long as residual < threshold
    - Stage 2: Full multi-wavelength unwrapping when Stage 1 residual is too high
    """

    def __init__(self,
                 lambda_1: float,
                 lambda_2: float,
                 max_capture: float = None,
                 confidence_threshold: float = 0.1,
                 outlier_residual_threshold: float = None,
                 edge_threshold: float = 0.5,
                 use_edge_detection: bool = True,
                 temporal_window_size: int = 4,
                 residual_threshold_fraction: float = 0.1,
                 # ===== Behavior Control Flags =====
                 unwrap_enabled: bool = True,
                 two_stage_enabled: bool = True,
                 temporal_filtering_mode: str = 'none',  # 'none', 'median', 'weighted_average'
                 target_device_idx=None,
                 precision=None):
        """
        Parameters
        ----------
        lambda_1, lambda_2 : float
            Wavelengths (must be different and positive)
        max_capture : float, optional
            Maximum capture range (default: synthetic_lambda / 2)
        confidence_threshold : float
            Confidence below this marks as outlier (default 0.1)
        outlier_residual_threshold : float, optional
            Residual threshold for outlier detection (default: lambda_2 / 8)
        edge_threshold : float
            Threshold for detecting discontinuities in microns (default 0.5)
        use_edge_detection : bool
            Enable edge detection to preserve sharp transitions (default True)
        temporal_window_size : int
            Number of frames for temporal filtering window (default 4)
        residual_threshold_fraction : float
            Stage 1 uses p1 while residual < fraction × lambda_1 (default 0.1)
            
        ===== BEHAVIOR CONTROL FLAGS (Easy Debugging) =====
        unwrap_enabled : bool
            If False: output = input (no processing, useful for testing)
            If True: apply unwrapping logic
        two_stage_enabled : bool
            If True: Use two-stage (Stage 1: direct p1, Stage 2: full unwrap)
            If False: Always perform full multi-wavelength unwrapping
        temporal_filtering_mode : str
            'none': No temporal filtering (recommended for closed-loop control)
            'median': Use median of last N frames (robust to outliers but adds lag)
            'weighted_average': Confidence-weighted average (trust good estimates)
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if lambda_1 <= 0 or lambda_2 <= 0:
            raise ValueError("Wavelengths must be positive")
        if abs(lambda_1 - lambda_2) < 1e-10:
            raise ValueError("Wavelengths must be different")

        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.confidence_threshold = float(confidence_threshold)
        self.edge_threshold = float(edge_threshold)
        self.use_edge_detection = use_edge_detection
        self.temporal_window_size = int(temporal_window_size)
        self.residual_threshold_fraction = float(residual_threshold_fraction)
        
        # ===== Behavior flags =====
        self.unwrap_enabled = bool(unwrap_enabled)
        self.two_stage_enabled = bool(two_stage_enabled)
        self.temporal_filtering_mode = str(temporal_filtering_mode).lower()
        
        if self.temporal_filtering_mode not in ('none', 'median', 'weighted_average'):
            raise ValueError(
                f"temporal_filtering_mode must be 'none', 'median', or 'weighted_average', "
                f"got '{temporal_filtering_mode}'"
            )

        # Compute synthetic wavelength and residual threshold for Stage 1 exit
        self.synthetic_lambda = (
            self.lambda_1 * self.lambda_2 / abs(self.lambda_1 - self.lambda_2)
        )
        # Residual threshold: when |residual| exceeds this, switch to Stage 2
        self.residual_threshold = residual_threshold_fraction * self.lambda_1

        if max_capture is None:
            self.max_capture = self.synthetic_lambda / 2.0
        else:
            self.max_capture = float(max_capture)

        if outlier_residual_threshold is None:
            self.outlier_residual_threshold = self.lambda_2 / 8.0
        else:
            self.outlier_residual_threshold = float(outlier_residual_threshold)

        self.max_k = int(self.xp.floor(self.max_capture / self.lambda_1)) - 1

        # Output arrays
        self.out_pistons_array = None
        self.out_pistonsU_array = None
        self.confidences = None
        self.outlier_flags = None
        self.residuals = None
        self.edge_flags = None
        self.stage_flags = None  # 0=no unwrap, 1=Stage 1, 2=Stage 2
        
        # Temporal history
        self.estimate_history = []
        self.confidence_history = []

        # Define inputs and outputs
        self.inputs['in_pistons_1'] = InputValue(type=BaseValue, optional=False)
        self.inputs['in_pistons_2'] = InputValue(type=BaseValue, optional=False)

        self.out_pistons_data = BaseValue(
            description='Unwrapped piston values from WLI phase unwrapping',
            target_device_idx=target_device_idx
        )
        self.out_pistonsU_data = BaseValue(
            description='Unwrapped piston values from WLI phase unwrapping',
            target_device_idx=target_device_idx
        )

        self.outputs['out_pistons'] = self.out_pistons_data
        self.outputs['out_pistonsU'] = self.out_pistonsU_data

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
            raise ValueError("Input shapes must match")

        nseg = p1_array.shape[0]

        self.out_pistons_array = self.xp.zeros(nseg, dtype=self.dtype)
        self.out_pistonsU_array = self.xp.zeros(nseg, dtype=self.dtype)
        self.confidences = self.xp.zeros(nseg, dtype=self.dtype)
        self.outlier_flags = self.xp.zeros(nseg, dtype=bool)
        self.residuals = self.xp.zeros(nseg, dtype=self.dtype)
        self.edge_flags = self.xp.zeros(nseg, dtype=bool)
        self.stage_flags = self.xp.zeros(nseg, dtype=self.xp.int32)

        self.estimate_history = []
        self.confidence_history = []
        self.out_pistons_data.value = self.out_pistons_array
        self.out_pistonsU_data.value = self.out_pistonsU_array

    @staticmethod
    def wrap_phase(x, wavelength):
        """Wrap phase values into [-wavelength/2, wavelength/2) interval."""
        return ((x + wavelength / 2.0) % wavelength) - wavelength / 2.0

    def _compute_confidence(self, estimate, p2_value):
        """
        Compute confidence score for an estimate by checking residual against p2.
        Used consistently for both Stage 1 and Stage 2.
        
        Returns
        -------
        confidence : float (0-1)
            Confidence score
        residual : float
            Residual error magnitude
        """
        wrapped_est = self.wrap_phase(estimate, self.lambda_2)
        residual = self.xp.abs(self.wrap_phase(wrapped_est - p2_value, self.lambda_2))
        
        # Confidence: 1.0 when residual=0, decays to 0 as residual approaches lambda_2/2
        confidence = self.xp.maximum(0.0, 1.0 - 2.0 * residual / self.lambda_2)
        
        return confidence, residual
    
    def _stage2_unwrap(self, p1_val, p2_val):
        """
        Perform full multi-wavelength unwrapping for a single piston.
        Generates candidates from p1 and finds best match to p2.
        
        Returns
        -------
        estimate, confidence, residual
        """
        k_values = self.xp.arange(-self.max_k, self.max_k + 1, dtype=self.xp.int32)
        candidates = p1_val + k_values * self.lambda_1
        wrapped_candidates = self.wrap_phase(candidates, self.lambda_2)
        
        residuals = self.xp.abs(wrapped_candidates - p2_val)
        
        # Filter by capture range and select best
        valid_mask = self.xp.abs(candidates) <= self.max_capture
        filtered_residuals = self.xp.where(valid_mask, residuals, self.xp.inf)
        best_idx = self.xp.argmin(filtered_residuals)
        estimate = candidates[best_idx]
        
        # Use SAME confidence computation as Stage 1 (for consistency)
        confidence, residual = self._compute_confidence(estimate, p2_val)
        
        return estimate, confidence, residual

    def _get_windowed_median(self, piston_idx):
        """
        Get median of estimates from temporal window.
        Robust to outliers.
        """
        if len(self.estimate_history) < 2:
            return None
        
        window_size = min(self.temporal_window_size, len(self.estimate_history))
        estimates = self.xp.array([h[piston_idx] for h in self.estimate_history[-window_size:]])
        
        return self.xp.median(estimates)

    def _get_windowed_weighted_average(self, piston_idx):
        """
        Get confidence-weighted average from temporal window.
        Higher confidence estimates have more influence.
        """
        if len(self.estimate_history) < 2:
            return None
        
        window_size = min(self.temporal_window_size, len(self.estimate_history))
        estimates = self.xp.array([h[piston_idx] for h in self.estimate_history[-window_size:]])
        confidences = self.xp.array([h[piston_idx] for h in self.confidence_history[-window_size:]])
        
        # Weight by confidence, avoid zero weights
        weights = self.xp.maximum(0.1, confidences)
        weighted_avg = self.xp.sum(estimates * weights) / self.xp.sum(weights)
        
        return weighted_avg

    def _apply_temporal_blending(self, i, estimate):
        """
        Apply temporal filtering based on configured mode.
        Properly blends current estimate with history (not just replacing).
        
        Returns
        -------
        blended_estimate : float
            Estimate blended with history (or original if no history)
        """
        if self.temporal_filtering_mode == 'none':
            return estimate
        
        if self.temporal_filtering_mode == 'median':
            windowed = self._get_windowed_median(i)
            if windowed is not None:
                # Blend: 70% current, 30% median of history
                # For control systems, prefer current for responsiveness
                return 0.7 * estimate + 0.3 * windowed
        
        elif self.temporal_filtering_mode == 'weighted_average':
            windowed = self._get_windowed_weighted_average(i)
            if windowed is not None:
                # Blend: 70% current, 30% weighted average of history
                return 0.7 * estimate + 0.3 * windowed
        
        return estimate

    def _detect_edges(self, estimates, nseg):
        """Detect abrupt changes (discontinuities) in the estimate field."""
        if nseg < 2 or not self.use_edge_detection:
            return self.xp.zeros(nseg, dtype=bool)

        edge_flags = self.xp.zeros(nseg, dtype=bool)
        diffs = self.xp.abs(self.xp.diff(estimates))
        large_jumps = diffs > self.edge_threshold
        
        edge_flags[:-1] = edge_flags[:-1] | large_jumps
        edge_flags[1:] = edge_flags[1:] | large_jumps
        
        return edge_flags

    def _apply_edge_preserving_filter(self, estimates, confidences, edge_flags, nseg):
        """Apply smoothing that respects edge boundaries."""
        if nseg < 3 or not self.use_edge_detection:
            return estimates

        smoothed = self.xp.copy(estimates)
        region_start = 0
        
        for i in range(1, nseg):
            if edge_flags[i]:
                region_length = i - region_start
                if region_length >= 3:
                    region = smoothed[region_start:i]
                    weights = self.xp.maximum(0.1, confidences[region_start:i])
                    weighted_vals = region * weights
                    smoothed[region_start:i] = weighted_vals / weights
                region_start = i
        
        # Process final region
        if region_start < nseg - 1:
            region_length = nseg - region_start
            if region_length >= 3:
                region = smoothed[region_start:]
                weights = self.xp.maximum(0.1, confidences[region_start:])
                weighted_vals = region * weights
                smoothed[region_start:] = weighted_vals / weights
        
        return smoothed

    def trigger_code(self):
        """
        Execute phase unwrapping with configurable behaviors.
        
        Optimized for closed-loop control:
        - Fast response to edges (minimal latency)
        - Correct Stage 1/Stage 2 switching based on residual
        - Proper temporal filtering (with blending, not replacement)
        
        Flow:
        1. If unwrap_enabled=False: output = input (no processing)
        2. If two_stage_enabled=True: Try Stage 1, switch to Stage 2 if residual too high
        3. If two_stage_enabled=False: Always perform full unwrapping
        4. Apply temporal filtering based on mode (optional, adds lag)
        5. Apply spatial filtering (edge-aware)
        """
        p1 = self.local_inputs['in_pistons_1'].get_value()
        p2 = self.local_inputs['in_pistons_2'].get_value()

        p1 = self.xp.asarray(p1, dtype=self.dtype)
        p2 = self.xp.asarray(p2, dtype=self.dtype)

        nseg = p1.shape[0]

        self.out_pistons_array[:] = p1

        # ===== PATH 1: No unwrapping (debug mode) =====
        if not self.unwrap_enabled:
            self.out_pistonsU_array[:] = p1
            self.confidences[:] = 1.0
            self.stage_flags[:] = 0  # No unwrapping
            self.residuals[:] = 0.0
            self.outlier_flags[:] = False
            return

        # ===== PATH 2: Two-stage or full unwrapping =====
        for i in range(nseg):
            if self.two_stage_enabled:
                # Try Stage 1: use p1 directly
                estimate = p1[i]
                confidence, residual = self._compute_confidence(estimate, p2[i])
                
                # Check if residual is too high → switch to Stage 2
                if residual > self.residual_threshold:
                    # Residual too high → switch to full unwrapping
                    estimate, confidence, residual = self._stage2_unwrap(p1[i], p2[i])
                    stage = 2
                else:
                    # Residual acceptable → stay with Stage 1
                    stage = 1
                    estimate = 0
            else:
                # Always do full unwrapping (Stage 2 only)
                estimate, confidence, residual = self._stage2_unwrap(p1[i], p2[i])
                stage = 2

            # Apply temporal blending (only if not 'none')
            if self.temporal_filtering_mode != 'none':
                estimate = self._apply_temporal_blending(i, estimate)

            # Store results
            self.out_pistonsU_array[i] = estimate
            self.confidences[i] = confidence
            self.residuals[i] = residual
            self.stage_flags[i] = stage
            self.outlier_flags[i] = (
                confidence < self.confidence_threshold or
                residual > self.outlier_residual_threshold
            )

        # ===== PATH 3: Edge detection and spatial filtering =====
        if self.use_edge_detection:
            self.edge_flags = self._detect_edges(self.out_pistonsU_array, nseg)

            smoothed = self._apply_edge_preserving_filter(
                self.out_pistonsU_array,
                self.confidences,
                self.edge_flags,
                nseg
            )
            
            # Replace outliers (that are not at edges) with smoothed values
            outlier_not_edge = self.outlier_flags & ~self.edge_flags
            self.out_pistonsU_array[:] = self.xp.where(
                outlier_not_edge,
                smoothed,
                self.out_pistonsU_array
            )
        else:
            self.edge_flags = self.xp.zeros(nseg, dtype=bool)

        # ===== PATH 4: Store in history for temporal filtering =====
        self.estimate_history.append(self.xp.copy(self.out_pistonsU_array))
        self.confidence_history.append(self.xp.copy(self.confidences))
        
        # Trim history to window size
        if len(self.estimate_history) > self.temporal_window_size:
            self.estimate_history.pop(0)
            self.confidence_history.pop(0)

    def post_trigger(self):
        """Set generation time and synchronize if using CUDA graphs."""
        super().post_trigger()
        self.out_pistons_data.generation_time = self.current_time
        self.out_pistonsU_data.generation_time = self.current_time

    @classmethod
    def input_names(cls):
        return {
            'in_pistons_1': (BaseValue, 'Piston values at wavelength lambda_1'),
            'in_pistons_2': (BaseValue, 'Piston values at wavelength lambda_2'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_pistons': (BaseValue, 'Piston values'),
            'out_pistonsU': (BaseValue, 'Unwrapped piston values'),
        }