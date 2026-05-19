import specula
specula.init(0)

import unittest
import numpy as np
from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.multirate_complementary_filter import MultirateComplementaryFilter
from test.specula_testlib import cpu_and_gpu

class MockSimulParams(SimulParams):
    def __init__(self, time_step=0.001):
        self.time_step = time_step

def build_double_integrator(g_f, target_device_idx, n_modes=1):
    num = np.array([[0.0, 0.0, g_f]])
    den = np.array([[1.0, -2.0, 1.0]])
    return IirFilterData([3], [3], num, den, n_modes=[n_modes], target_device_idx=target_device_idx)

class TestMultirateFilter(unittest.TestCase):

    @cpu_and_gpu
    def test_sync_validation_errors(self, target_device_idx, xp):
        """Test that the synchronization check properly raises RuntimeErrors."""
        engine = build_double_integrator(0.1, target_device_idx)
        sp = MockSimulParams()

        filt = MultirateComplementaryFilter(sp, engine, g_track=0.1, weights=[0.5, 0.5], N_list=[3],
                                            target_device_idx=target_device_idx)

        v_yf = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)
        v_ys = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)

        filt.inputs['in_yf'].set(v_yf)
        filt.inputs['in_ys'].set([v_ys])
        filt.local_inputs['in_yf'] = filt.inputs['in_yf'].get(target_device_idx)
        filt.local_inputs['in_ys'] = filt.inputs['in_ys'].get(target_device_idx)
        filt.setup()

        # Frame 1: Not a slow frame (N=3). Slow sensor SHOULD have an old generation time.
        v_yf.generation_time = 0.001
        v_ys.generation_time = 0.000 # Old time! Correct!
        filt.check_ready(0.001)
        filt.trigger_code()
        filt.post_trigger()

        # Frame 2: Let's artificially break sync. Frame 2 is NOT a slow frame,
        # but we pretend the slow sensor just arrived.
        v_yf.generation_time = 0.002
        v_ys.generation_time = 0.002 # NEW time! Wrong!
        with self.assertRaisesRegex(RuntimeError, "updated unexpectedly"):
            filt.check_ready(0.002)

        # Let's fix it for Frame 2
        v_ys.generation_time = 0.000
        filt.check_ready(0.002)
        filt.trigger_code()
        filt.post_trigger()

        # Frame 3: Is a slow frame (N=3). Slow sensor MUST have the new generation time.
        # Let's pretend it didn't arrive.
        v_yf.generation_time = 0.003
        v_ys.generation_time = 0.000 # OLD time! Wrong!
        with self.assertRaisesRegex(RuntimeError, "missing update"):
            filt.check_ready(0.003)

    @cpu_and_gpu
    def test_zero_stuffed_inputs_can_skip_sync_validation(self, target_device_idx, xp):
        """Slow inputs coming from an upstream zero-stuffing stage may update every frame."""
        engine = build_double_integrator(0.1, target_device_idx)
        sp = MockSimulParams()

        filt = MultirateComplementaryFilter(
            sp, engine, g_track=0.1, weights=[1/3, 1/3, 1/3], N_list=[2, 2],
            validate_sync=False, target_device_idx=target_device_idx
        )

        v_yf = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)
        v_ys1 = BaseValue(value=np.array([0.0]), target_device_idx=target_device_idx)
        v_ys2 = BaseValue(value=np.array([0.0]), target_device_idx=target_device_idx)

        filt.inputs['in_yf'].set(v_yf)
        filt.inputs['in_ys'].set([v_ys1, v_ys2])
        filt.local_inputs['in_yf'] = filt.inputs['in_yf'].get(target_device_idx)
        filt.local_inputs['in_ys'] = filt.inputs['in_ys'].get(target_device_idx)
        filt.setup()

        for k in range(4):
            t_sim = (k + 1) * 0.001
            v_yf.generation_time = t_sim
            v_ys1.generation_time = t_sim
            v_ys2.generation_time = t_sim
            filt.check_ready(t_sim)
            filt.trigger_code()
            filt.post_trigger()

        self.assertEqual(filt.out_comm.generation_time, 0.004)

    @cpu_and_gpu
    def test_basic_1_fast_1_slow_with_lifecycle(self, target_device_idx, xp):
        """Test exact mathematical correctness respecting the SPECULA trigger lifecycle."""
        g_f = 0.1
        g_track = 0.1
        weights = [0.5, 0.5]
        N_list = [3]

        engine = build_double_integrator(g_f, target_device_idx)
        sp = MockSimulParams()
        filt = MultirateComplementaryFilter(sp, engine, g_track, weights, N_list,
                                            target_device_idx=target_device_idx)

        v_yf = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)
        v_ys = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)
        v_ys.generation_time = -1 # Initial valid time

        filt.inputs['in_yf'].set(v_yf)
        filt.inputs['in_ys'].set([v_ys])
        filt.local_inputs['in_yf'] = filt.inputs['in_yf'].get(target_device_idx)
        filt.local_inputs['in_ys'] = filt.inputs['in_ys'].get(target_device_idx)
        filt.setup()

        n_steps = 15
        out_comm_sim = np.zeros(n_steps)

        for k in range(n_steps):
            t_sim = (k + 1) * 0.001
            v_yf.generation_time = t_sim

            # Sync Logic for Tests: Update slow sensor only on Nth frames
            if (k + 1) % N_list[0] == 0:
                v_ys.generation_time = t_sim

            filt.check_ready(t_sim)
            filt.trigger_code()
            filt.post_trigger()

            out_comm_sim[k] = cpuArray(filt.out_comm.get_value())[0]

    @cpu_and_gpu
    def test_setup_validation(self, target_device_idx, xp):
        """Test validation of inputs and parameter lengths."""

        sp = MockSimulParams()

        # 1. Test mismatched parameter lists (weights deve essere N_list + 1)
        with self.assertRaises(ValueError):
            MultirateComplementaryFilter(sp, None, g_track=0.1, weights=[0.5, 0.5], N_list=[5, 10],
                                         target_device_idx=target_device_idx)

        # 2. Test mismatched connected inputs
        engine = build_double_integrator(0.1, target_device_idx)
        filt = MultirateComplementaryFilter(sp,engine, g_track=0.1, weights=[0.5, 0.5], N_list=[5],
                                            target_device_idx=target_device_idx)

        # Connect yf but leave ys empty
        v_yf = BaseValue(value=np.zeros(1), target_device_idx=target_device_idx)
        filt.inputs['in_yf'].set(v_yf)
        filt.inputs['in_ys'].set([])  # Expected 1 slow sensor, got 0

        # Mock the get_all_inputs phase
        filt.local_inputs['in_yf'] = filt.inputs['in_yf'].get(target_device_idx)
        filt.local_inputs['in_ys'] = filt.inputs['in_ys'].get(target_device_idx)

        with self.assertRaises(ValueError):
            filt.setup()

    @cpu_and_gpu
    def test_basic_1_fast_1_slow(self, target_device_idx, xp):
        """Test exact mathematical correctness for 1 fast and 1 slow sensor."""
        g_f = 0.1
        g_track = 0.1
        weights = [0.5, 0.5]  # Baricentro al 50%
        N_list = [3]

        engine = build_double_integrator(g_f, target_device_idx)
        sp = MockSimulParams()
        filt = MultirateComplementaryFilter(sp, engine, g_track, weights, N_list,
                                            target_device_idx=target_device_idx)

        # Initialize inputs
        v_yf = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)
        v_ys = BaseValue(value=np.array([1.0]), target_device_idx=target_device_idx)

        filt.inputs['in_yf'].set(v_yf)
        filt.inputs['in_ys'].set([v_ys])

        filt.local_inputs['in_yf'] = filt.inputs['in_yf'].get(target_device_idx)
        filt.local_inputs['in_ys'] = filt.inputs['in_ys'].get(target_device_idx)
        filt.setup()

        n_steps = 15
        out_comm_sim = np.zeros(n_steps)

        # Run the filter
        for k in range(n_steps):
            time = (k + 1) * 0.001
            v_yf.generation_time = time
            if (k + 1) % N_list[0] == 0:
                v_ys.generation_time = time
            filt.check_ready(time)
            filt.trigger_code()
            filt.post_trigger()
            out_comm_sim[k] = cpuArray(filt.out_comm.get_value())[0]

        # Expected manual calculation (Pure Python difference equation)
        out_comm_expected = np.zeros(n_steps)
        yf_prev = 0.0

        w_fast = 0.5
        w_slow = 0.5
        c_yf_0 = 1.0 + (g_track * w_fast)
        c_yf_1 = -1.0
        c_ys_0 = g_track * w_slow * N_list[0]

        for k in range(n_steps):
            frame = k + 1  # GPU frame_counter pre-increments

            yf = 1.0
            ys = 1.0
            ys_stuffed = ys if (frame % N_list[0] == 0) else 0.0

            mixed = (c_yf_0 * yf) + (c_yf_1 * yf_prev) + (c_ys_0 * ys_stuffed)
            yf_prev = yf

            # Double integrator: u[k] = 2u[k-1] - u[k-2] + g_f * mixed
            u_m1 = out_comm_expected[k-1] if k >= 1 else 0.0
            u_m2 = out_comm_expected[k-2] if k >= 2 else 0.0

            out_comm_expected[k] = 2 * u_m1 - u_m2 + g_f * mixed

        # Assert absolute match between C++/CUDA implementation and math
        np.testing.assert_allclose(out_comm_sim, out_comm_expected, rtol=1e-6, atol=1e-6,
                                   err_msg="The CUDA graph multirate implementation"
                                           " does not match the LTI difference equation.")

    @cpu_and_gpu
    def test_vector_input_routing(self, target_device_idx, xp):
        """Test that the vector input routing produces the exact same result as separate inputs."""
        g_f = 0.1
        g_track = 0.1
        weights = [1/3, 1/3, 1/3]
        N_list = [4, 10]
        sp = MockSimulParams()

        # 1. Setup Filter A (Separate Inputs)
        # Pass n_modes=2 to ensure the filter can handle vector inputs of size 2 for both yf and ys.
        engine_a = build_double_integrator(g_f, target_device_idx, n_modes=2)
        filt_a = MultirateComplementaryFilter(sp, engine_a, g_track, weights, N_list,
                                              target_device_idx=target_device_idx)

        v_yf = BaseValue(value=np.array([1.5, 2.5]), target_device_idx=target_device_idx)
        v_ys1 = BaseValue(value=np.array([-1.0, -1.0]), target_device_idx=target_device_idx)
        v_ys2 = BaseValue(value=np.array([0.5, -0.5]), target_device_idx=target_device_idx)
        v_ys1.generation_time = -1  # Must differ from fast sensor's initial time
        v_ys2.generation_time = -1  # Must differ from fast sensor's initial time

        filt_a.inputs['in_yf'].set(v_yf)
        filt_a.inputs['in_ys'].set([v_ys1, v_ys2])
        filt_a.local_inputs['in_yf'] = filt_a.inputs['in_yf'].get(target_device_idx)
        filt_a.local_inputs['in_ys'] = filt_a.inputs['in_ys'].get(target_device_idx)
        filt_a.setup()

        # 2. Setup Filter B (Vector Input)
        # Vector is: [ yf_0, yf_1, ys1_0, ys1_1, ys2_0, ys2_1 ]
        v_vec = BaseValue(value=np.array([1.5, 2.5, -1.0, -1.0, 0.5, -0.5]),
                          target_device_idx=target_device_idx)

        # PASSAGGIAMO n_modes=2 ANCHE QUI!
        engine_b = build_double_integrator(g_f, target_device_idx, n_modes=2)
        filt_b = MultirateComplementaryFilter(
            sp, engine_b, g_track, weights, N_list,
            idx_yf=[0, 1],               # Indices for yf
            idx_ys=[[2, 3], [4, 5]],     # Indices for ys1 and ys2
            target_device_idx=target_device_idx
        )

        filt_b.inputs['in_vec'].set(v_vec)
        filt_b.local_inputs['in_vec'] = filt_b.inputs['in_vec'].get(target_device_idx)
        # Must mock the get_all_inputs for the optional ones so they are None
        filt_b.local_inputs['in_yf'] = None
        filt_b.setup()

        # Run both for a few steps
        for i in range(15):
            time = (i + 1) * 0.001
            v_yf.generation_time = time
            if (i + 1) % N_list[0] == 0:
                v_ys1.generation_time = time
            if (i + 1) % N_list[1] == 0:
                v_ys2.generation_time = time
            filt_a.prepare_trigger(time)
            filt_a.trigger_code()
            filt_b.prepare_trigger(time)
            filt_b.trigger_code()

            out_a = cpuArray(filt_a.out_comm.get_value())
            out_b = cpuArray(filt_b.out_comm.get_value())

            # The outputs should be absolutely identical
            np.testing.assert_array_equal(out_a, out_b)

    @cpu_and_gpu
    def test_morfeo_3_ngs_case(self, target_device_idx, xp):
        """Test advanced 3-NGS case (1 fast, 2 slow at different framerates) with debug plotting."""

        g_f = 0.20
        g_track = 0.005
        # Weights for the "Tripletta" case: 1 fast sensor and 2 slow sensors with
        # equal influenceon the barycenter.
        weights = [1/3, 1/3, 1/3]
        N_list = [4, 10]

        engine = build_double_integrator(g_f, target_device_idx)
        sp = MockSimulParams()
        filt = MultirateComplementaryFilter(sp, engine, g_track, weights, N_list,
                                            target_device_idx=target_device_idx)

        # Connect Inputs
        v_yf = BaseValue(value=np.array([0.0]), target_device_idx=target_device_idx)
        v_ys1 = BaseValue(value=np.array([0.0]), target_device_idx=target_device_idx)
        v_ys2 = BaseValue(value=np.array([0.0]), target_device_idx=target_device_idx)
        v_ys1.generation_time = -1  # Must differ from fast sensor's initial time
        v_ys2.generation_time = -1  # Must differ from fast sensor's initial time

        filt.inputs['in_yf'].set(v_yf)
        filt.inputs['in_ys'].set([v_ys1, v_ys2])

        filt.local_inputs['in_yf'] = filt.inputs['in_yf'].get(target_device_idx)
        filt.local_inputs['in_ys'] = filt.inputs['in_ys'].get(target_device_idx)
        filt.setup()

        # Simulation settings (Closed-loop)
        n_steps = 2000
        f_fast = 1000.0
        t = np.arange(n_steps) / f_fast

        # Sinusoidal reference with offset
        f_rif = 2.0
        R = np.sin(2 * np.pi * f_rif * t) + 1.0

        # Biased readings but with total arithmetic sum ZERO.
        # Given the "Tripletta" setup, the expected final error is -(B_f + B_s1 + B_s2)/3 = 0.
        B_f = 2.0
        B_s1 = -5.0
        B_s2 = 3.0

        x_true = np.zeros(n_steps)
        u_true = np.zeros(n_steps)
        err_tracking_true = np.zeros(n_steps)

        yf_hist = np.zeros(n_steps)
        ys1_hist = np.zeros(n_steps)
        ys2_hist = np.zeros(n_steps)

        tau = 0.002
        alpha = np.exp(-1.0 / (f_fast * tau))

        for k in range(1, n_steps):
            time = (k + 1) * 0.001

            # Plant dynamics
            x_true[k] = alpha * x_true[k-1] + (1 - alpha) * u_true[k-1]

            # True physical error (delayed by 1 frame due to sensor read)
            e_true = R[k] - x_true[k-1]

            # Differential measurements with extreme static biases
            err_f = e_true + B_f
            err_s1 = e_true + B_s1
            err_s2 = e_true + B_s2

            v_yf.set_value(np.array([err_f]))
            v_ys1.set_value(np.array([err_s1]))
            v_ys2.set_value(np.array([err_s2]))
            v_yf.generation_time = time
            if k % N_list[0] == 0:
                v_ys1.generation_time = time
            if k % N_list[1] == 0:
                v_ys2.generation_time = time

            yf_hist[k] = err_f
            ys1_hist[k] = err_s1
            ys2_hist[k] = err_s2

            filt.check_ready(time)
            filt.trigger_code()
            filt.post_trigger()
            u_true[k] = cpuArray(filt.out_comm.get_value())[0]

            # Store true instantaneous tracking error
            err_tracking_true[k] = R[k] - x_true[k]

        # Change this to True to see the plots locally during debugging!
        debug_plot = False
        if debug_plot: # pragma: no cover
            self._debug_plot_morfeo_case(t, R, x_true, err_tracking_true,
                                         yf_hist, ys1_hist, ys2_hist, u_true, N_list,
                                         B_f, B_s1, B_s2)

        # Basic check: control effort should be completely stable
        self.assertTrue(np.all(np.isfinite(u_true)),
                        "Control effort diverged to infinity/NaN!")

        # The true tracking error should converge near 0 despite the massive biases
        # We check the max error in the last 200 samples (steady state)
        max_ss_error = np.max(np.abs(err_tracking_true[-200:]))
        self.assertLess(max_ss_error, 0.1, f"Failed to reject bias."
                        f" Max steady-state error: {max_ss_error}")

    def _debug_plot_morfeo_case(self, t, R, x_true, err_true, yf_hist,
                                ys1_hist, ys2_hist, u_true, N_list, B_f,
                                B_s1, B_s2): # pragma: no cover
        """Helper to plot the MORFEO 3-NGS simulation results."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 12))

        # PLOT 1: Tracking Performance
        plt.subplot(3, 1, 1)
        plt.plot(t, R, 'k--', label='Target Reference $R(t)$', linewidth=2)
        plt.plot(t, x_true, 'b-', label='Plant Output $x(t)$', alpha=0.8)
        plt.title('Sinusoidal Tracking Performance (Tripletta 3-NGS Multirate Fusion)')
        plt.ylabel('Position')
        plt.legend(loc='lower right')
        plt.grid(True)

        # PLOT 2: Sensor Biases and True Error
        plt.subplot(3, 1, 2)
        # Simulate ZOH visual for the slow sensors
        s1_zoh = np.zeros_like(ys1_hist)
        s2_zoh = np.zeros_like(ys2_hist)
        for k in range(len(t)):
            frame = k + 1
            s1_zoh[k] = ys1_hist[k] if (frame % N_list[0] == 0) else s1_zoh[k-1]
            s2_zoh[k] = ys2_hist[k] if (frame % N_list[1] == 0) else s2_zoh[k-1]

        plt.plot(t, yf_hist, 'c', alpha=0.5, label=f'Fast Sensor Reading (Bias {B_f})')
        plt.plot(t, s1_zoh, 'r', alpha=0.5, label=f'Slow Sensor 1 ZOH (Bias {B_s1}, N={N_list[0]})')
        plt.plot(t, s2_zoh, 'g', alpha=0.5, label=f'Slow Sensor 2 ZOH (Bias {B_s2}, N={N_list[1]})')
        plt.plot(t, err_true, 'k-', linewidth=2, label='True Tracking Error ($R - x$)')
        plt.title('Sensor Readings vs True Physical Error (Barycentric Rejection)')
        plt.ylabel('Tracking Error')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.ylim(-6, 4)

        # PLOT 3: Control Action
        plt.subplot(3, 1, 3)
        plt.plot(t, u_true, 'k-', linewidth=1.5, label='Total Control Action $U(z)$')
        plt.title('Multirate Double Integrator Control Effort')
        plt.xlabel('Time [s]')
        plt.ylabel('Actuator Command')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
