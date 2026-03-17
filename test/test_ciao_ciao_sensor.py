import unittest

import specula
specula.init(0)

import numpy as np

from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.lib.compute_petal_ifunc import compute_petal_ifunc
from specula.processing_objects.ciao_ciao_sensor import CiaoCiaoSensor
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
        ef.generation_time = t

        wfs.inputs['in_ef'].set(ef)
        wfs.setup()
        wfs.check_ready(t)
        wfs.trigger()
        wfs.post_trigger()

        out_i = wfs.outputs['out_i']

        self.assertEqual(out_i.i.shape, (number_px, number_px))
        np.testing.assert_allclose(
            cpuArray(xp.sum(out_i.i)),
            cpuArray(ref_S0 * ef.masked_area()),
            rtol=1e-7,
            atol=0.0
        )

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
        ef.generation_time = t

        wfs.inputs['in_ef'].set(ef)
        wfs.setup()
        wfs.check_ready(t)
        wfs.trigger()
        wfs.post_trigger()

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
        ef.generation_time = t

        wfs.inputs['in_ef'].set(ef)
        wfs.setup()
        wfs.check_ready(t)
        wfs.trigger()
        wfs.post_trigger()

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
        ef.generation_time = t

        wfs.inputs['in_ef'].set(ef)
        wfs.setup()
        wfs.check_ready(t)
        wfs.trigger()
        wfs.post_trigger()

        out = cpuArray(wfs.outputs['out_i'].i)
        cos_delta = np.clip(out / 2.0 - 1.0, -1.0, 1.0)
        measured_delta = np.arccos(cos_delta)

        sector_means = []
        for sector in sector_masks:
            sector_cpu = cpuArray(sector) > 0.5
            sector_means.append(np.mean(measured_delta[sector_cpu]))
        sector_means = np.asarray(sector_means)

        np.testing.assert_allclose(sector_means, 0.0, atol=1e-3)
