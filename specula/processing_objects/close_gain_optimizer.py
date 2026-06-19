import numpy as np
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue


class CloseGainOptimizer(BaseProcessingObj):
    """
    CLOSE (Correlation-Locked Optimization StratEgy) gain optimizer processing object.
    
    Implements a self-regulating tracker for modal integrator based AO loops, updating
    modal gains in real-time based on the temporal auto-correlation of modal measurements.
    Implements the correlation-locking approach: see Equations (3) and (4) in "CLOSE: a
    self-regulating, best-performance tracker for modal integrator based AO loops",
    Deo et al. (2019).
    
    Parameters
    ----------
    nmodes : int [1]
        Number of modes to optimize. Defines the dimensionality of the gain vector
        and the size of the modal measurement vector.
    initial_gain : float [1], optional
        Initial value for all modal gains. Default: 0.5
    dt : float [1], optional
        Time-shift (in frames) at which correlation is evaluated. 
        Should be 2xdelay + 1, where delay is the pure delay of the control
        (total delay = delay + 1 frame). Default is 3.0 (delay = 1.0).
    p : float [1], optional
        Low-pass filter coefficient for autocorrelation estimators (Equation 3).
        Should be in range (0, 1]. Default: 0.3.
    r : float [1], optional
        Target correlation ratio setpoint. Defines the desired normalized autocorrelation
        value at lag dt. Theoretical value should be 0.0. Default: -0.1
    q_plus : float [1], optional
        Tracking gain increase factor. Learning rate for positive correlation error
        (when correlation is above setpoint). Controls how aggressively gains increase
        during normal operation. Should be small (typically 1e-2 to 1e-1). Default: 1e-2
    q_minus_ratio : float [1], optional
        Ratio of q_minus to q_plus for aggressive correction. When correlation is below
        setpoint (indicating ringing/overshoot), gain adjustment uses q_minus = q_plus * q_minus_ratio.
        Typically > 1 for faster damping. Default: 5.0
    target_device_idx : int [1], optional
        Target device index for computation (e.g., GPU device number).
        If None, uses CPU or default device. Default: None
    precision : int [1], optional
        Numerical precision for computations (e.g., 32 for float32, 64 for float64).
        If None, uses default precision. Default: None
    """

    def __init__(self,
                 nmodes: int,
                 dt: float = 3,     
                 initial_gain: float = 0.5, 
                 p: float = 0.3,
                 r: float = -0.1,          
                 q_plus: float = 1e-2,        # Tracking gain increase factor
                 q_minus_ratio: float = 5.0,  # Ratio of q- to q+ (for overshoots)
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.p = p
        self.r = r
        self.dt = int(np.ceil(dt))
        self.dt_frac = self.dt - dt
        
        # Determine the asymmetric learning factors q+ and q-
        self.q_plus = q_plus
        self.q_minus = q_plus * q_minus_ratio

        # History buffer to store m_i[k] up to k - dt
        self.m_history = [] 

        # State variables for Equation 3 (Estimators)
        self.N_0 = self.xp.zeros(nmodes, dtype=self.dtype)
        self.N_dt = self.xp.zeros(nmodes, dtype=self.dtype)

        # Output state for Equation 4
        self.optimized_gain = BaseValue(
            value=self.xp.ones(self.nmodes, dtype=self.dtype) * initial_gain,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Inputs & Outputs definition
        self.inputs['in_modes'] = InputValue(type=BaseValue)
        self.outputs['out_gains'] = self.optimized_gain

    @classmethod
    def input_names(cls):
        return {'in_modes': InputDesc(BaseValue, 'Input modal measurement vector (m_i[k])')}

    @classmethod
    def output_names(cls):
        return {'out_gains': OutputDesc(BaseValue, 'Optimized modal gain vector (G_i[k])')}

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        # Fetch the current modal measurements
        self.current_m_i = self.local_inputs['in_modes'].value

    def trigger_code(self):
        # 1. Update the history buffer
        self.m_history.append(self.current_m_i.copy())
        if len(self.m_history) > self.dt + 1:
            self.m_history.pop(0)

        # 2. Wait until we have enough history to compute the dt-shifted correlation
        if len(self.m_history) == self.dt + 1:
            m_k = self.current_m_i
            m_k_dt = (1.0 - self.dt_frac) * self.m_history[0] + self.dt_frac * self.m_history[1] 
            # m_k_dt = self.m_history[0]  # The measurement from \Delta t frames ago

            # 3. Equation 3 (Deo et al. 2019): Autocorrelation estimators
            self.N_0 = self.p * (m_k ** 2) + (1 - self.p) * self.N_0
            self.N_dt = self.p * (m_k * m_k_dt) + (1 - self.p) * self.N_dt

            safe_N_0 = self.xp.where(self.N_0 < 1e-12, 1e-12, self.N_0) # avoid division by 0
            correlation_ratio = self.N_dt / safe_N_0

            # 4. Equation 4 (Deo et al. 2019): Gain update
            corr_diff = correlation_ratio - self.r

            # Asymmetry application: 
            # If the normalized correlation is less than the setpoint r, it implies 
            # ringing/overshooting, so we use the more aggressive q_minus. Otherwise, q_plus.
            q_array = self.xp.where(corr_diff < 0, self.q_minus, self.q_plus)

            current_gain = self.optimized_gain.value
            new_gain = current_gain * (1.0 + q_array * corr_diff)

            # Safety clamp: Ensure gains don't drop to 0 or explode during massive transients
            new_gain = self.xp.clip(new_gain, 1e-4, 10.0)

            # Store updated gains
            self.optimized_gain.value[:] = new_gain

    def post_trigger(self):
        super().post_trigger()
        # Update generation time for down-stream processing
        self.optimized_gain.generation_time = self.current_time
