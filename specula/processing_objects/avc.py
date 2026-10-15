from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams


class AVC(BaseProcessingObj):
    """
    Adaptive Vibration Cancellation

    Cancels a narrowband vibration with a lock-in integrator and a frequency
    tracking loop (PLL), given a fixed calibration of the plant at the
    vibration frequency. Fixed-plant variant of the algorithm of Muradore,
    Pettazzi & Fedrigo (ECC 2014).

    **The algorithm.** Per sample, with ``c = cos(alpha)``, ``s = sin(alpha)``
    and ``mu = gx * Ts``::

        u  = thetac * c + thetas * s          # command, added to the loop
        y  = in_measurement                   # residual of the mode
        x3 += mu * c * y                      # integrate, in the rotating
        x4 += mu * s * y                      #   frame (a lock-in)
        omega -= 2 * gomega * Ts * s * y      # frequency tracking (PLL)
        alpha += Ts * omega - 2 * k * gomega * Ts * s * y

    Multiplying the residual by ``(cos alpha, sin alpha)`` demodulates it into the frame
    rotating at the vibration frequency, and accumulating integrates it
    there; in complex form the two state lines are simply
    ``Lambda_hat += mu * exp(i*alpha) * y``.

    Because the plant is constant, the averaged dynamics is a *linear time-invariant* scalar complex system,
    ``d(Lambda_hat)/dt = gamma * (Lambda_star - kappa * Lambda_hat)`` with
    ``gamma = gx/2`` and ``kappa = conj(P_star / P_hat)``. Hence, globally:

    - a unique equilibrium, at which the tone is cancelled exactly;
    - exponential stability **if and only if**
      ``|arg(P_star) - arg(P_hat)| < 90 deg``;  <-- the stability cone
    - a closed-form rate, ``0.5 * gx * Re(kappa)``;
    - a step-size limit ``gx * Ts < 2 * Re(kappa) / |kappa|^2``, typically
      three orders of magnitude above the gain actually used.

    Note that what converges is the residual, not an identification:
    ``Lambda_hat`` settles on whatever value cancels the tone *given* the
    calibration in use, and equals the true disturbance only if the
    calibration is exact.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters; only ``time_step`` (the AO loop period Ts)
        is used.
    n_avc : int
        Number of independent instances (vibrations). ``in_measurement``
        and ``out_comm`` have this length. Several instances may be fed the
        same measurement, and their corrections summed downstream (e.g.
        with :py:class:`~specula.processing_objects.linear_combination.LinearCombination`).
    freq : float or array-like [Hz], length n_avc
        Initial guess of the vibration frequency. The PLL will pull it to
        the true value provided the tone dominates the residual near it.
    gx : float or array-like, length n_avc
        Gain of the disturbance integrator. The convergence rate is
        ``0.5 * gx * Re(kappa)`` [1/s]; see the step-size limit above.
    gomega : float or array-like, length n_avc
        Gain of the frequency loop. Set to 0 to hold the frequency fixed at
        ``freq``. Must be re-tuned when the tone's signal-to-noise ratio
        changes, since the PLL is driven by whatever dominates the error.
    k : float or array-like, length n_avc
        Phase/frequency coupling in the PLL (proportional path).
    plant_re, plant_im : float or array-like, length n_avc, optional
        Calibrated closed-loop plant response at the vibration frequency,
        in cartesian form: ``P_hat = plant_re + 1j * plant_im``. Defaults
        to ``1 + 0j``. The phase must be within 90 degrees of the true
        one, the magnitude can be any non-zero value. See
        :py:mod:`specula.lib.avc_plant_model` for an analytic estimate.
    plant_gain, plant_phase : float or array-like, length n_avc, optional
        The same calibration in polar form, ``|P_hat|`` and
        ``arg(P_hat)`` in **degrees**. Mutually exclusive with
        ``plant_re``/``plant_im``; provided because the calibration is
        normally measured as a magnitude and a phase.
    alpha0 : float or array-like, length n_avc, optional
        Initial phase estimate [rad]. Default 0.
    target_device_idx : int, optional
    precision : int, optional

    Inputs
    ------
    in_measurement : BaseValue, length n_avc
        Current measurement y[k] (e.g. the residual of one modal
        coefficient).

    Outputs
    -------
    out_comm : BaseValue, length n_avc
        Correction u[k], computed at the previous iteration, ready to be
        added to the loop command.
    out_freq : BaseValue, length n_avc
        Current frequency estimate [Hz].
    out_state : BaseValue, shape (n_avc, 4)
        ``[plant_re, plant_im, x3, x4]``. The first two are the frozen
        calibration and never change
    """

    def __init__(self,
                 simul_params: SimulParams,
                 n_avc: int,
                 freq,
                 gx,
                 gomega,
                 k,
                 plant_re=None,
                 plant_im=None,
                 plant_gain=None,
                 plant_phase=None,
                 alpha0=0.0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if n_avc < 1:
            raise ValueError('n_avc must be at least 1')
        self.n_avc = n_avc
        self.T = float(simul_params.time_step)

        # Tuning parameters, broadcast to n_avc elements if scalar
        self.gx = self.to_param(gx, 'gx')
        self.gomega = self.to_param(gomega, 'gomega')
        self.k = self.to_param(k, 'k')

        cartesian = (plant_re is not None) or (plant_im is not None)
        polar = (plant_gain is not None) or (plant_phase is not None)
        if cartesian and polar:
            raise ValueError(
                'Give the plant calibration either as plant_re/plant_im or as '
                'plant_gain/plant_phase, not both')
        if polar:
            rho = self.to_param(1.0 if plant_gain is None else plant_gain, 'plant_gain')
            psi = self.to_param(0.0 if plant_phase is None else plant_phase, 'plant_phase')
            psi = psi * (self.xp.pi / 180.0)
            p_re = rho * self.xp.cos(psi)
            p_im = rho * self.xp.sin(psi)
        else:
            p_re = self.to_param(1.0 if plant_re is None else plant_re, 'plant_re')
            p_im = self.to_param(0.0 if plant_im is None else plant_im, 'plant_im')

        energy = p_re * p_re + p_im * p_im
        if bool(self.xp.any(energy <= 0)):
            raise ValueError(
                'The plant calibration must be non-zero: plant_re and plant_im '
                'cannot both be zero for any AVC instance')
        self.plant_re = p_re
        self.plant_im = p_im

        # Fixed map (x3, x4) -> (thetac, thetas). With
        # theta = -Lambda_hat / conj(P_hat) = -(1/rho) exp(i psi) Lambda_hat,
        # this is a rotation by the plant phase and a scaling by the inverse
        # plant gain, i.e.
        #     thetac = -( x3 cos psi - x4 sin psi) / rho
        #     thetas = -( x3 sin psi + x4 cos psi) / rho
        # which, written in cartesian form, is -(P_re, P_im)/|P|^2 contracted
        # with (x3, x4). Evaluated once here; no division at run time.
        self.m11 = -p_re / energy
        self.m12 = p_im / energy
        self.m21 = -p_im / energy
        self.m22 = -p_re / energy

        # Only the disturbance quadratures and the frequency loop. alpha_c
        # is the Kahan compensation carried by the phase accumulator.
        freq_arr = self.to_param(freq, 'freq')
        self.x3 = self.xp.zeros(n_avc, dtype=self.dtype)
        self.x4 = self.xp.zeros(n_avc, dtype=self.dtype)
        self.omega = self.xp.array(2 * self.xp.pi * freq_arr, dtype=self.dtype)
        self.alpha = self.to_param(alpha0, 'alpha0')
        self.alpha_c = self.xp.zeros(n_avc, dtype=self.dtype)

        # Correction u[k]; zero at start since x3 = x4 = 0
        self.correction = self.xp.zeros(n_avc, dtype=self.dtype)

        # Working buffers
        self.measurement = self.xp.zeros(n_avc, dtype=self.dtype)
        self.out_buffer = self.xp.zeros(n_avc, dtype=self.dtype)
        self.freq_buffer = self.xp.zeros(n_avc, dtype=self.dtype)
        self.state_buffer = self.xp.zeros((n_avc, 4), dtype=self.dtype)
        self.state_buffer[:, 0] = self.plant_re
        self.state_buffer[:, 1] = self.plant_im

        # Outputs
        self.out_comm = BaseValue(
            'AVC correction command',
            value=self.xp.zeros(n_avc, dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision)
        self.out_freq = BaseValue(
            'AVC vibration frequency estimate [Hz]',
            value=self.xp.zeros(n_avc, dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision)
        self.out_state = BaseValue(
            'AVC state [plant_re, plant_im, x3, x4] (diagnostic)',
            value=self.xp.zeros((n_avc, 4), dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision)

        self.inputs['in_measurement'] = InputValue(type=BaseValue)
        self.outputs['out_comm'] = self.out_comm
        self.outputs['out_freq'] = self.out_freq
        self.outputs['out_state'] = self.out_state

    def to_param(self, value, name):
        """Convert a scalar or array-like parameter to a static xp array of
        length n_avc, broadcasting scalars. Only used at init time."""
        arr = self.xp.atleast_1d(self.xp.asarray(value, dtype=self.dtype))
        if arr.size == 1:
            arr = self.xp.full(self.n_avc, arr.reshape(-1)[0], dtype=self.dtype)
        elif arr.size != self.n_avc:
            raise ValueError(
                f"Parameter '{name}' has size {arr.size} but n_avc={self.n_avc}")
        return arr

    @classmethod
    def input_names(cls):
        return {'in_measurement': InputDesc(BaseValue, 'Input measurement signal, one value per AVC instance')}

    @classmethod
    def output_names(cls):
        return {
            'out_comm': OutputDesc(BaseValue, 'Output correction command, one value per AVC instance'),
            'out_freq': OutputDesc(BaseValue, 'Estimated vibration frequency [Hz], one value per AVC instance'),
            'out_state': OutputDesc(BaseValue, 'State [plant_re, plant_im, x3, x4] per AVC instance'),
        }

    def setup(self):
        super().setup()
        self.build_stream()

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        measurement = self.local_inputs['in_measurement'].value
        measurement_array = self.xp.asarray(measurement, dtype=self.dtype)
        measurement_array = self.xp.atleast_1d(measurement_array).ravel()

        if measurement_array.size != self.n_avc:
            if measurement_array.size == 1:
                measurement_array = self.xp.full(
                    self.n_avc, measurement_array[0], dtype=self.dtype)
            else:
                raise ValueError(
                    f"Input in_measurement has size {measurement_array.size} "
                    f"but AVC expects {self.n_avc} values")

        self.measurement[:] = measurement_array

    def trigger_code(self):
        xp = self.xp

        # Step 1: emit the correction computed at the previous iteration,
        # and the current frequency estimate.
        self.out_buffer[:] = self.correction
        self.freq_buffer[:] = self.omega / (2 * xp.pi)

        alpha = self.alpha
        omega = self.omega
        alpha_c = self.alpha_c
        cos_a = xp.cos(alpha)
        sin_a = xp.sin(alpha)

        # Step 2: the measured residual.
        y = self.measurement

        # Step 3: integrate the residual in the frame rotating at the
        # estimated vibration frequency. Multiplying by (cos, sin) is the
        # demodulation, accumulating is the integration; together they are
        # Lambda_hat += mu * exp(i alpha) * y.
        mu = self.gx * self.T
        new_x3 = self.x3 + mu * cos_a * y
        new_x4 = self.x4 + mu * sin_a * y
        self.x3[:] = new_x3
        self.x4[:] = new_x4
        self.state_buffer[:, 2] = new_x3
        self.state_buffer[:, 3] = new_x4

        # Step 4: frequency loop (PLL). The phase accumulator uses Kahan
        # compensated summation, since it runs for the whole simulation and
        # a naive sum drifts.
        alpha_input = (self.T * omega -
                       2.0 * self.k * self.T * self.gomega * sin_a * y)
        alpha_y = alpha_input - alpha_c
        alpha_t = alpha + alpha_y
        self.alpha_c[:] = (alpha_t - alpha) - alpha_y
        alpha_new = alpha_t
        alpha_new = xp.where(alpha_new > xp.pi, alpha_new - 2 * xp.pi, alpha_new)
        alpha_new = xp.where(alpha_new < -xp.pi, alpha_new + 2 * xp.pi, alpha_new)
        self.alpha[:] = alpha_new

        # Frequency update (uses alpha from before this iteration's update)
        self.omega[:] = omega - 2.0 * self.T * self.gomega * sin_a * y

        # Step 5: refresh the control quadratures through the fixed map, and
        # form the correction for the next iteration.
        thetac = self.m11 * new_x3 + self.m12 * new_x4
        thetas = self.m21 * new_x3 + self.m22 * new_x4
        self.correction[:] = xp.cos(alpha_new) * thetac + xp.sin(alpha_new) * thetas

    def post_trigger(self):
        super().post_trigger()
        self.out_comm.value[:] = self.out_buffer
        self.out_comm.generation_time = self.current_time
        self.out_freq.value[:] = self.freq_buffer
        self.out_freq.generation_time = self.current_time
        self.out_state.value[:] = self.state_buffer
        self.out_state.generation_time = self.current_time
