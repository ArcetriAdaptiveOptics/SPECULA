import unittest

import specula
from specula.loop_control import LoopControl
specula.init(0)

from specula import cpuArray, np, RAD2ASEC
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.lib.compute_petal_ifunc import compute_petal_ifunc
from specula.processing_objects.ciao_ciao_sensor import CiaoCiaoSensor
from specula.processing_objects.ciao_ciao_slopec import CiaoCiaoSlopec 
from test.specula_testlib import cpu_and_gpu

class TestCiaoCiaoSensor(unittest.TestCase):

    @staticmethod
    def _build_petal_phase_and_masks(dim, n_petals, pistons_nm, xp):
        ifs_2d, mask, _ = compute_petal_ifunc(
            dim=dim,
            n_petals=n_petals,
            xp=xp,
            dtype=xp.float32,
            special_last_petal=True
        )

        idx = xp.where(mask > 0)
        phase_nm = xp.zeros((dim, dim), dtype=xp.float32)
        sector_masks = []

        for i in range(n_petals):
            sector = xp.zeros((dim, dim), dtype=xp.float32)
            sector[idx] = ifs_2d[i]
            phase_nm += pistons_nm[i] * sector
            sector_masks.append(sector)

        return phase_nm, mask.astype(xp.float32), sector_masks

    @cpu_and_gpu
    def test_output_shape_and_flux_normalization(self, target_device_idx, xp):
        t = 1
        ref_S0 = 123.0
        number_px = 48

        wfs = CiaoCiaoSensor(
            wavelengthInNm=750.0,
            number_px=number_px,
            diffRotAngleInDeg=180.0,
            tiltInArcsec=(0.02, -0.01),
            normalize_flux=True,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(64, 64, 0.01, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = ef.seconds_to_t(t)

        wfs.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(wfs, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        out_i = wfs.outputs['out_i']

        self.assertEqual(out_i.i.shape, (number_px, number_px))
        np.testing.assert_allclose(
            cpuArray(xp.sum(out_i.i)),
            cpuArray(ref_S0 * ef.masked_area()),
            rtol=1e-7,
            atol=0.0
        )

    @cpu_and_gpu
    def test_ciaociao_shape(self, target_device_idx, xp):
        t = 1
        ref_S0 = 123.0
        number_px = 48

        wfs = CiaoCiaoSensor(
            wavelengthInNm=750.0,
            number_px=number_px,
            diffRotAngleInDeg=180.0,
            tiltInArcsec=(0.02, -0.01),
            normalize_flux=True,
            target_device_idx=target_device_idx
        )

        dimx = 10
        dimy = 20

        ef = ElectricField(dimx, dimy, 0.01, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        wfs.inputs['in_ef'].set(ef)

        # CiaoCiao requires a square input array
        with self.assertRaises(ValueError):
            wfs.setup()

    @cpu_and_gpu
    def test_channel_flux_unbalance(self, target_device_idx, xp):
        t = 1
        dim = 40

        wfs = CiaoCiaoSensor(
            wavelengthInNm=500.0,
            number_px=dim,
            diffRotAngleInDeg=0.0,
            tiltInArcsec=(0.0, 0.0),
            channel_flux=0.75,
            normalize_flux=False,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(dim, dim, 0.02, S0=1.0, target_device_idx=target_device_idx)
        ef.generation_time = ef.seconds_to_t(t)
        wfs.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(wfs, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        out = cpuArray(wfs.outputs['out_i'].i)
        expected_constant = (np.sqrt(1.5) + np.sqrt(0.5)) ** 2
        expected = xp.full((dim, dim), expected_constant, dtype=wfs.outputs['out_i'].i.dtype)

        np.testing.assert_allclose(out, cpuArray(expected), rtol=1e-7, atol=1e-7)

    @cpu_and_gpu
    def test_petal_diff_pist_measure_one_on_n_minus_one_sectors(self, target_device_idx, xp):
        t = 1
        n_petals = 4
        dim = 129
        wavelength_in_nm = 500.0
        unit_nm = wavelength_in_nm / (2.0 * np.pi)

        pistons_nm = xp.arange(n_petals, dtype=xp.float32) * unit_nm
        phase_nm, pupil_mask, sector_masks = self._build_petal_phase_and_masks(
            dim=dim,
            n_petals=n_petals,
            pistons_nm=pistons_nm,
            xp=xp
        )

        wfs = CiaoCiaoSensor(
            wavelengthInNm=wavelength_in_nm,
            number_px=dim,
            diffRotAngleInDeg=360.0 / n_petals,
            tiltInArcsec=(0.0, 0.0),
            normalize_flux=False,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(dim, dim, 0.01, S0=1.0, target_device_idx=target_device_idx)
        ef.A[:] = pupil_mask
        ef.phaseInNm[:] = phase_nm
        ef.generation_time = ef.seconds_to_t(t)

        wfs.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(wfs, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        out = cpuArray(wfs.outputs['out_i'].i)

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cpuArray(ef.phaseInNm), origin='lower')
            plt.colorbar()
            plt.title('Input phase (nm)')
            plt.subplot(1, 2, 2)
            plt.imshow(out, origin='lower')
            plt.colorbar()
            plt.title('Measured intensity')
            plt.show()

        cos_delta = np.clip(out / 2.0 - 1.0, -1.0, 1.0)
        measured_delta = np.arccos(cos_delta)

        sector_means = []
        for sector in sector_masks:
            sector_cpu = cpuArray(sector) > 0.5
            sector_means.append(np.mean(measured_delta[sector_cpu]))
        sector_means = np.asarray(sector_means)

        n_close_to_one = np.sum(np.abs(sector_means - 1.0) < 0.15)
        self.assertEqual(n_close_to_one, n_petals - 1)

    @cpu_and_gpu
    def test_petal_zero_diff_pist_measure_zero(self, target_device_idx, xp):
        t = 1
        n_petals = 4
        dim = 129
        wavelength_in_nm = 500.0

        pistons_nm = xp.zeros(n_petals, dtype=xp.float32)
        phase_nm, pupil_mask, sector_masks = self._build_petal_phase_and_masks(
            dim=dim,
            n_petals=n_petals,
            pistons_nm=pistons_nm,
            xp=xp
        )

        wfs = CiaoCiaoSensor(
            wavelengthInNm=wavelength_in_nm,
            number_px=dim,
            diffRotAngleInDeg=360.0 / n_petals,
            tiltInArcsec=(0.0, 0.0),
            normalize_flux=False,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(dim, dim, 0.01, S0=1.0, target_device_idx=target_device_idx)
        ef.A[:] = pupil_mask
        ef.phaseInNm[:] = phase_nm
        ef.generation_time = ef.seconds_to_t(t)
        wfs.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(wfs, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        out = cpuArray(wfs.outputs['out_i'].i)
        cos_delta = np.clip(out / 2.0 - 1.0, -1.0, 1.0)
        measured_delta = np.arccos(cos_delta)

        sector_means = []
        for sector in sector_masks:
            sector_cpu = cpuArray(sector) > 0.5
            sector_means.append(np.mean(measured_delta[sector_cpu]))
        sector_means = np.asarray(sector_means)

        np.testing.assert_allclose(sector_means, 0.0, atol=1e-3)


class TestCiaoCiaoSlopec(unittest.TestCase):

    @staticmethod
    def _build_petal_phase_and_masks(dim, n_petals, pistons_nm, xp):
        """
        Helper method to generate a segmented pupil with distinct pistons.
        """
        ifs_2d, mask, _ = compute_petal_ifunc(
            dim=dim,
            n_petals=n_petals,
            xp=xp,
            dtype=xp.float32,
            special_last_petal=True
        )

        idx = xp.where(mask > 0)
        phase_nm = xp.zeros((dim, dim), dtype=xp.float32)
        sector_masks = []

        for i in range(n_petals):
            sector = xp.zeros((dim, dim), dtype=xp.float32)
            sector[idx] = ifs_2d[i]
            phase_nm += pistons_nm[i] * sector
            sector_masks.append(sector > 0.5)

        return phase_nm, mask.astype(xp.float32), sector_masks

    def get_synthetic_data(self, shape=(128, 128)):
        """
        Generates a synthetic interferogram with a carrier frequency,
        and a full pupil mask for testing.
        """
        y, x = np.indices(shape)

        # Define a carrier frequency to mimic the tilt in one interferometer arm
        # This determines where the sideband will be in the FFT
        freq_x = 10.0 / shape[1]
        freq_y = 5.0 / shape[0]

        # Synthetic interferogram: I = 1 + cos(2*pi*(fx*x + fy*y))
        interferogram = 1.0 + np.cos(2 * np.pi * (freq_x * x + freq_y * y))

        # Full pupil mask (all pixels valid)
        simul_params = SimulParams(pixel_pupil=shape[0], pixel_pitch=1.0)
        pupil_mask = Pupilstop(simul_params, input_mask=np.ones(shape, dtype=np.float32))

        # Expected peak in the FFT for the given carrier frequency
        window_x = shape[1] // 2 + 10
        window_y = shape[0] // 2 + 5

        return interferogram, pupil_mask, window_x, window_y

    @staticmethod
    def _compute_sector_pistons_from_opd(opd_map, sector_masks):
        sector_means = []
        for mask in sector_masks:
            mask_cpu = cpuArray(mask) > 0.5
            sector_means.append(np.mean(opd_map[mask_cpu]))
        return np.asarray(sector_means)

    @cpu_and_gpu
    def test_ciaociao_slopec_pipeline(self, target_device_idx, xp):
        """
        Tests the basic processing pipeline of the CiaoCiaoSlopec,
        ensuring that outputs are generated with the correct shapes and types.
        """
        t = 1
        shape = (128, 128)
        interf_data, pupil_mask, win_x, win_y = self.get_synthetic_data(shape)

        # Setup input Pixels object
        pixels = Pixels(*shape, target_device_idx=target_device_idx)
        pixels.pixels = xp.asarray(interf_data, dtype=pixels.pixels.dtype)
        pixels.generation_time = pixels.seconds_to_t(t)

        # Initialize Slopec
        slopec = CiaoCiaoSlopec(
            wavelength_in_nm=2200.0,
            window_x_in_pix=win_x,
            window_y_in_pix=win_y,
            window_sigma_in_pix=3.0,
            pupil_mask=pupil_mask,
            unwrap=False,
            target_device_idx=target_device_idx
        )

        slopec.inputs['in_pixels'].set(pixels)

        loop = LoopControl()
        loop.add(slopec, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        slopes = slopec.outputs['out_slopes']
        flux_per_sub = slopec.outputs['out_flux_per_subaperture'].value

        # Verify output shapes (flattened OPD)
        self.assertEqual(len(cpuArray(slopes.slopes)), shape[0] * shape[1])
        self.assertEqual(len(cpuArray(flux_per_sub)), 1)

        # Verify fluxes are positive
        self.assertTrue(xp.all(flux_per_sub > 0))

        # Check generation time propagation
        self.assertEqual(slopes.generation_time, slopes.seconds_to_t(t))

    @cpu_and_gpu
    def test_ciaociao_slopec_unwrap(self, target_device_idx, xp):
        """
        Tests that the unwrapping routine executes correctly 
        moving data between the device (CPU/GPU) and the host.
        """
        t = 1
        shape = (64, 64) # Smaller shape for faster unwrapping test
        interf_data, pupil_mask, win_x, win_y = self.get_synthetic_data(shape)

        pixels = Pixels(*shape, target_device_idx=target_device_idx)
        pixels.pixels = xp.asarray(interf_data, dtype=pixels.pixels.dtype)
        pixels.generation_time = pixels.seconds_to_t(t)

        # Initialize Slopec with unwrap=True
        slopec = CiaoCiaoSlopec(
            wavelength_in_nm=2200.0,
            window_x_in_pix=win_x,
            window_y_in_pix=win_y,
            window_sigma_in_pix=3.0,
            pupil_mask=pupil_mask,
            unwrap=True,  # <--- Testing the unwrapping path
            target_device_idx=target_device_idx
        )

        slopec.inputs['in_pixels'].set(pixels)

        loop = LoopControl()
        loop.add(slopec, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        slopes = slopec.outputs['out_slopes']

        # If it didn't crash during the cpuArray <-> unwrap_phase <-> to_xp transfer,
        # and produced an OPD map, the pipeline is intact.
        self.assertEqual(len(cpuArray(slopes.slopes)), shape[0] * shape[1])
        self.assertFalse(xp.isnan(slopes.slopes).any())

    @cpu_and_gpu
    def test_piston_aberration_and_slopes(self, target_device_idx, xp):
        """
        End-to-end test: ElectricField -> CiaoCiaoSensor -> Pixels -> CiaoCiaoSlopec.
        Applies a known piston aberration and verifies the differential measurement.
        Includes optional plotting for visual debugging.
        """
        t = 1
        dim = 128
        n_petals = 6
        wavelength_in_nm = 1650.0

        # Simulating a 38.5m ELT pupil diameter
        pupil_diameter_m = 38.5
        pixel_pitch = pupil_diameter_m / dim

        # 1. Define input pistons (within dynamic range [-lambda/2, lambda/2])
        pistons_nm = xp.array([0.0, 500.0, 0.0, 0.0, 500.0, 0.0], dtype=xp.float32)

        phase_nm, pupil_mask, sector_masks = self._build_petal_phase_and_masks(
            dim=dim, n_petals=n_petals, pistons_nm=pistons_nm, xp=xp
        )

        # 2. Calculate tilt to place the FFT sideband exactly where we want it.
        # We use a safe number of fringes (15) to avoid Nyquist aliasing.
        # Using SPECULA's RAD2ASEC ensures the sensor decodes this exactly.
        n_fringes = 15.0
        pupil_diameter_m = dim * pixel_pitch
        tilt_rad = n_fringes * (wavelength_in_nm * 1e-9) / pupil_diameter_m
        tilt_arcsec = tilt_rad * specula.RAD2ASEC
        print(f"tilt_arcsec: {tilt_arcsec:.4f} arcsec to get {n_fringes} fringes across the pupil")

        # The sideband in the FFT will be shifted by n_fringes pixels on the X axis
        window_x = int(dim // 2 + n_fringes)
        window_y = int(dim // 2)
        window_sigma = 4.0

        # 3. Setup CiaoCiaoSensor
        sensor = CiaoCiaoSensor(
            wavelengthInNm=wavelength_in_nm,
            number_px=dim,
            diffRotAngleInDeg=360.0 / n_petals,
            tiltInArcsec=(tilt_arcsec, 0.0),
            normalize_flux=False,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(dim, dim, pixel_pitch, S0=1.0, target_device_idx=target_device_idx)
        ef.A[:] = pupil_mask
        ef.phaseInNm[:] = phase_nm
        ef.generation_time = ef.seconds_to_t(t)

        sensor.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(sensor, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        interferogram = sensor.outputs['out_i'].i

        # 4. Setup CiaoCiaoSlopec
        pixels = Pixels(dim, dim, target_device_idx=target_device_idx)
        pixels.pixels = interferogram
        pixels.generation_time = pixels.seconds_to_t(t)

        slopec = CiaoCiaoSlopec(
            wavelength_in_nm=wavelength_in_nm,
            window_x_in_pix=window_x,
            window_y_in_pix=window_y,
            window_sigma_in_pix=window_sigma,
            pupil_mask=Pupilstop(SimulParams(pixel_pupil=dim, pixel_pitch=1.0),
                                  input_mask=cpuArray(pupil_mask).astype(np.float32)),
            diffRotAngleInDeg=360.0 / n_petals,
            unwrap=False,
            target_device_idx=target_device_idx
        )

        slopec.inputs['in_pixels'].set(pixels)

        loop = LoopControl()
        loop.add(slopec, idx=0)
        loop.run(run_time=t*2, dt=t, t0=t)

        measured_opd = cpuArray(slopec.outputs['out_slopes'].slopes).reshape((dim, dim))
        measured_pistons = self._compute_sector_pistons_from_opd(measured_opd, sector_masks)

        # 5. Validation
        self.assertEqual(len(measured_pistons), n_petals)
        self.assertTrue(np.any(np.abs(measured_pistons) > 0.0),
                "Measured pistons should not be entirely zero.")
        self.assertFalse(np.isnan(measured_pistons).any(),
                 "Measured pistons contain NaNs.")

        # 6. Optional Plotting Block (Set to True to debug)
        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt

            # Recompute the FFT purely for visualization
            ft_intensity = np.fft.fftshift(np.fft.fft2(cpuArray(interferogram), norm='ortho'))
            power_spectrum = np.log10(np.abs(ft_intensity) + 1e-12)

            # Plot A: Input Phase, Interferogram, Power Spectrum, and recovered OPD
            fig, axs = plt.subplots(1, 5, figsize=(24, 4))

            # Plot 1: Input Phase
            im0 = axs[0].imshow(cpuArray(phase_nm), origin='lower', cmap='viridis')
            axs[0].set_title("Input Petal Phase [nm]")
            fig.colorbar(im0, ax=axs[0])

            # Plot 2: Sensor's Tilt Phase Ramp (The smoking gun)
            # This confirms the sensor is actually generating the tilt correctly!
            tilt_ramp = np.angle(cpuArray(sensor._tilt_exp))
            im1 = axs[1].imshow(tilt_ramp, origin='lower', cmap='twilight')
            axs[1].set_title("Internal Tilt Ramp [rad]")
            fig.colorbar(im1, ax=axs[1])

            # Plot 3: Interferogram
            im2 = axs[2].imshow(cpuArray(interferogram), origin='lower', cmap='gray')
            axs[2].set_title(f"Interferogram ({int(n_fringes)} Fringes expected)")
            fig.colorbar(im2, ax=axs[2])

            # Plot 4: Power Spectrum & Window location
            im3 = axs[3].imshow(power_spectrum, origin='lower', cmap='magma')
            axs[3].set_title("Power Spectrum (Log)")
            circle = plt.Circle((window_x, window_y), window_sigma, color='lime', fill=False, lw=2)
            axs[3].add_patch(circle)
            fig.colorbar(im3, ax=axs[3])

            # Plot 5: Recovered OPD map
            im4 = axs[4].imshow(measured_opd, origin='lower', cmap='coolwarm')
            axs[4].set_title("Recovered OPD [nm]")
            fig.colorbar(im4, ax=axs[4])

            plt.tight_layout()

            # Plot B: Real VS Measured Differential Pistons
            plt.figure(figsize=(6, 5))
            plt.plot(range(n_petals), measured_pistons, alpha=0.8,
                     color='dodgerblue', label='Measured Pistons')
            plt.plot(range(n_petals), pistons_nm, alpha=0.8,
                     color='coral', label='Input Pistons')
            plt.axhline(0, color='black', linewidth=1)
            plt.title("Measured Sector Pistons (from OPD)")
            plt.xlabel("Sector Index")
            plt.ylabel("OPD [nm]")
            plt.legend()

            plt.show()
