import os
import unittest
import numpy as np

import specula
specula.init(0, 1)  # Default target device, single precision

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.petal_unwrapper import PetalUnwrapper
from test.specula_testlib import cpu_and_gpu

class TestPetalUnwrapperRealBasis(unittest.TestCase):

    @cpu_and_gpu
    def test_andes_basis_robustness(self, target_device_idx, xp):
        # 1. Load the real modal basis
        fits_path = '/raid2/gcarla/git/ANDES/andes/PASSATA_scripts/data/ifunc/ANDES_400pix_79nacts_cir__5zern_10000.0cn_OOMAO_hexObstr.fits'

        if not os.path.exists(fits_path):
            self.skipTest(f"File {fits_path} not found. Please update the path.")

        ifunc = IFunc.restore(fits_path, target_device_idx=target_device_idx)

        # Extrapolate parameters
        dim = 400
        pixel_pitch = 39.0 / dim  # Assuming 39m ELT, adjust if needed
        n_petals = 6
        n_petal_modes = 10

        simul_params = SimulParams(time_step=1, pixel_pupil=dim, pixel_pitch=pixel_pitch)
        pupilstop = Pupilstop(simul_params, input_mask=cpuArray(ifunc.mask_inf_func),
                              target_device_idx=target_device_idx)

        # 2. Unwrapper Initialization parameters
        thresh = 350.0
        angle_offset = 30.0  # <--- THIS MIGHT BE THE CULPRIT (Try 0.0 or 90.0 if the plot looks wrong)
        spider_widths = [0.5] * n_petals # <--- INCREASE THIS IF MASKS FALL IN THE SHADOW (e.g. 0.8)
        
        unwrapper = PetalUnwrapper(
            ifunc=ifunc,
            pupilstop=pupilstop,
            n_petals=n_petals,
            angle_offset_deg=angle_offset,
            spider_widths=spider_widths,
            thresh_in_nm=thresh,
            n_petal_modes=n_petal_modes,
            target_device_idx=target_device_idx
        )

        # --- CRITICAL NUMERICAL DIAGNOSTICS ---
        H_cpu = cpuArray(unwrapper.H)
        H_cond = np.linalg.cond(H_cpu)
        print(f"\n[DEBUG] H matrix shape: {H_cpu.shape}")
        print(f"[DEBUG] H Condition Number: {H_cond:.2e}")

        _, S, _ = np.linalg.svd(H_cpu)
        print(f"[DEBUG] Singular Values of H: {S}")

        # --- VISUAL DEBUGGING ---
        debug_plot = True
        if debug_plot:
            import matplotlib.pyplot as plt

            mask_amp = cpuArray(pupilstop.A) > 0
            x_1d = (np.arange(dim) - dim / 2.0 + 0.5) * pixel_pitch
            X, Y = np.meshgrid(x_1d, x_1d)

            fig, axs = plt.subplots(1, 3, figsize=(16, 5))

            # Plot 1: Pupil & Spider Geometry
            axs[0].imshow(mask_amp, origin='lower', extent=[x_1d[0], x_1d[-1], x_1d[0], x_1d[-1]], cmap='gray')
            for i in range(n_petals):
                theta_rad = np.radians(angle_offset + i * (360.0 / n_petals))
                # Draw a line from center outwards
                axs[0].plot([0, 19.5 * np.cos(theta_rad)], [0, 19.5 * np.sin(theta_rad)], 'r-', lw=2)
            axs[0].set_title(f"Geometry (Offset: {angle_offset}°)")

            # Plot 2: Mask evaluation for Spider 0
            theta_rad = np.radians(angle_offset)
            nx, ny = -np.sin(theta_rad), np.cos(theta_rad)
            D = X * nx + Y * ny
            margin = spider_widths[0] * 1.5

            # Highlight left and right strips
            eval_mask = np.zeros((dim, dim))
            eval_mask[(D < 0) & (D > -margin) & mask_amp] = -1 # Left = Blue
            eval_mask[(D >= 0) & (D < margin) & mask_amp] = 1  # Right = Red

            axs[1].imshow(eval_mask, origin='lower', cmap='coolwarm', vmin=-1, vmax=1)
            axs[1].set_title("Sampling Masks for Spider 0")

            # Plot 3: The very last mode of the basis
            last_mode_2d = np.zeros((dim, dim))
            last_mode_2d[mask_amp] = cpuArray(ifunc.influence_function[-1, :])
            axs[2].imshow(last_mode_2d, origin='lower', cmap='viridis')
            axs[2].set_title("Basis Mode: index -1")

            plt.tight_layout()
            plt.show()

        # Fail explicitly if ill-conditioned (after showing the plot)
        self.assertLess(H_cond, 1e4, "H matrix is severely ill-conditioned! Check the plot to fix alignment.")

        # --- 3. Monte Carlo Test (Will run only if assertion passes) ---
        n_modes_total = ifunc.influence_function.shape[0]
        n_atmo_modes = n_modes_total - n_petal_modes

        for test_idx in range(3):
            # A) Random atmosphere
            atmo_comm = xp.random.randn(n_atmo_modes, dtype=xp.float32) * 200.0

            # B) Background petal noise (below threshold)
            petal_comm = xp.random.randn(n_petal_modes, dtype=xp.float32) * 50.0

            # C) Force an extreme error on a random spider gap
            target_petal_idx = np.random.randint(0, n_petals)
            h_err_target = np.zeros(12, dtype=np.float32)
            h_err_target[2 * target_petal_idx] = 800.0     # Inner edge jump
            h_err_target[2 * target_petal_idx + 1] = 800.0 # Outer edge jump

            # Convert gap jump to modal command using pseudo-inverse
            m_jump = cpuArray(unwrapper.H_dagger) @ h_err_target
            petal_comm += xp.array(m_jump, dtype=xp.float32)

            in_comm_data = xp.concatenate((atmo_comm, petal_comm))
            in_comm = BaseValue(value=in_comm_data, target_device_idx=target_device_idx)
            in_comm.generation_time = test_idx + 1

            unwrapper.inputs['in_comm'].set(in_comm)
            unwrapper.setup()
            unwrapper.check_ready(test_idx + 1)
            unwrapper.trigger()
            unwrapper.post_trigger()

            out_comm = unwrapper.outputs['out_comm'].value

            # Verification 1: Atmosphere must remain untouched
            xp.testing.assert_allclose(
                out_comm[:n_atmo_modes],
                atmo_comm,
                rtol=1e-5,
                err_msg=f"Test {test_idx}: Filter corrupted the atmospheric modes!"
            )

            # Verification 2: Petal errors must be reduced below threshold
            m_pet_out = out_comm[-n_petal_modes:]
            final_gaps = unwrapper.H @ m_pet_out

            max_gap = float(xp.max(xp.abs(final_gaps)))
            self.assertLess(
                max_gap,
                thresh,
                f"Test {test_idx}: Unwrapping failed! Residual gap {max_gap:.1f} nm > {thresh} nm"
            )
