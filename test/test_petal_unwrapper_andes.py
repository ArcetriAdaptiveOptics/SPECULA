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

        simul_params = SimulParams(time_step=1, pixel_pupil=dim, pixel_pitch=pixel_pitch)
        pupilstop = Pupilstop(simul_params, input_mask=cpuArray(ifunc.mask_inf_func),
                              target_device_idx=target_device_idx)

        # 2. Unwrapper Initialization parameters
        thresh = 350.0
        angle_offset = 30.0  # <--- Adjust if the red lines in the plot don't match the gaps
        spider_widths = [0.5] * n_petals # <--- Increase if masks fall into the shadow

        unwrapper = PetalUnwrapper(
            ifunc=ifunc,
            pupilstop=pupilstop,
            n_petals=n_petals,
            angle_offset_deg=angle_offset,
            spider_widths=spider_widths,
            thresh_in_nm=thresh,
            target_device_idx=target_device_idx
        )

        # --- CRITICAL NUMERICAL DIAGNOSTICS ---
        # We now check H_full (shape: 12 x N_modes)
        H_full_cpu = cpuArray(unwrapper.H_full)
        print(f"\n[DEBUG] H_full matrix shape: {H_full_cpu.shape}")

        # Look at the singular values to ensure the 12 gaps are linearly independent
        _, S, _ = np.linalg.svd(H_full_cpu, full_matrices=False)
        print(f"[DEBUG] Singular Values of H_full (top 12): {S[:12]}")

        # --- VISUAL DEBUGGING ---
        debug_plot = False
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

            # Plot 3: The synthesized Petal 0
            # This proves that the agnostic algorithm successfully built a petal out of your basis!
            petal_0_comm = cpuArray(unwrapper.C_petals)[:, 0]
            petal_0_2d = np.zeros((dim, dim))
            B_cpu = cpuArray(ifunc.influence_function)
            petal_0_2d[mask_amp] = petal_0_comm @ B_cpu
            
            axs[2].imshow(petal_0_2d, origin='lower', cmap='viridis')
            axs[2].set_title("Synthesized Petal 0 from Basis")

            plt.tight_layout()
            plt.show()

        # Check that the mapping pseudo-inverse isn't blowing up
        H_pd_cpu = cpuArray(unwrapper.H_petals_dagger)
        self.assertFalse(np.isnan(H_pd_cpu).any(),
                         "H_petals_dagger contains NaNs. Geometry mapping failed.")

        # --- 3. Monte Carlo Test ---
        n_modes_total = ifunc.influence_function.shape[0]

        # Enable the second debug plot to see the action
        debug_plot_mc = False

        for test_idx in range(3):
            # A) Base command simulating standard atmospheric correction
            atmo_comm = (xp.random.randn(n_modes_total) * 20.0).astype(xp.float32)

            # B) Force a PHYSICAL error: Piston one random petal by 800 nm
            target_petal_idx = np.random.randint(0, n_petals)

            ideal_petal_amplitudes = np.zeros(n_petals * 3, dtype=np.float32)
            ideal_petal_amplitudes[3 * target_petal_idx + 0] = 800.0  # +0 is Piston, +1 is Tip, +2 is Tilt

            # Map the ideal petal to the actuator basis
            delta_comm = cpuArray(unwrapper.C_petals) @ ideal_petal_amplitudes

            in_comm_data = atmo_comm + xp.array(delta_comm, dtype=xp.float32)

            in_comm = BaseValue(value=in_comm_data, target_device_idx=target_device_idx)
            in_comm.generation_time = test_idx + 1

            unwrapper.inputs['in_comm'].set(in_comm)
            unwrapper.setup()
            unwrapper.check_ready(test_idx + 1)
            unwrapper.trigger()
            unwrapper.post_trigger()

            out_comm = unwrapper.outputs['out_comm'].value

            # --- PLOTTING THE ACTION (Only for the first test iteration) ---
            if debug_plot_mc and test_idx == 0:
                import matplotlib.pyplot as plt

                # We need the basis to reconstruct 2D phases
                B_cpu = cpuArray(ifunc.influence_function)

                # Reconstruct 2D Phase Maps
                phase_in = np.zeros((dim, dim))
                phase_in[mask_amp] = cpuArray(in_comm_data) @ B_cpu

                phase_out = np.zeros((dim, dim))
                phase_out[mask_amp] = cpuArray(out_comm) @ B_cpu

                phase_petal = np.zeros((dim, dim))
                phase_petal[mask_amp] = delta_comm @ B_cpu

                # Get the Jumps (Gaps) before and after
                gaps_before = cpuArray(unwrapper.H_full) @ cpuArray(in_comm_data)
                gaps_after = cpuArray(unwrapper.H_full) @ cpuArray(out_comm)

                fig2, axs2 = plt.subplots(2, 2, figsize=(16, 12))

                # Top Left: The injected Petal Error
                im0 = axs2[0, 0].imshow(phase_petal, origin='lower',
                                        cmap='seismic', vmin=-1000, vmax=1000)
                axs2[0, 0].set_title(f"Injected Pure Petal Piston (Petal {target_petal_idx})")
                fig2.colorbar(im0, ax=axs2[0, 0])

                # Top Right: Total Input Phase (Atmo + Petal)
                im1 = axs2[0, 1].imshow(phase_in, origin='lower', cmap='viridis')
                axs2[0, 1].set_title("Input Phase (Atmosphere + Petal Error)")
                fig2.colorbar(im1, ax=axs2[0, 1])

                # Bottom Left: Output Phase (Cleaned)
                im2 = axs2[1, 0].imshow(phase_out, origin='lower', cmap='viridis')
                axs2[1, 0].set_title("Output Phase (Unwrapped)")
                fig2.colorbar(im2, ax=axs2[1, 0])

                # Bottom Right: Bar chart of the 12 Jumps
                x_pos = np.arange(12)
                axs2[1, 1].bar(x_pos - 0.2, gaps_before, 0.4, label='Gaps Before', color='red')
                axs2[1, 1].bar(x_pos + 0.2, gaps_after, 0.4, label='Gaps After', color='green')
                axs2[1, 1].axhline(thresh, color='k', linestyle='--', label='+Threshold')
                axs2[1, 1].axhline(-thresh, color='k', linestyle='--', label='-Threshold')
                axs2[1, 1].set_title("Physical Gaps measured by H_full")
                axs2[1, 1].set_xticks(x_pos)
                axs2[1, 1].set_xlabel("Measurement Point Index (0-11)")
                axs2[1, 1].set_ylabel("Gap [nm]")
                axs2[1, 1].legend()

                plt.tight_layout()
                plt.show()

            # Verification: Petal errors must be reduced below threshold
            final_gaps = unwrapper.H_full @ out_comm
            max_gap = float(xp.max(xp.abs(final_gaps)))

            self.assertLess(
                max_gap,
                thresh,
                f"Test {test_idx}: Unwrapping failed! Residual gap {max_gap:.1f} nm > {thresh} nm"
            )
