"""
Tests of the LGS geometry: sub-aperture centres and WFS rotation of the launcher.

- calc_noise_cov_elong() uses the sub-aperture centres, so for a launcher at
  the pupil centre the elongation is mirror-symmetric, and it can rotate the
  launcher from the telescope frame to the WFS frame (wfs_rotation_in_deg).
- launcher_in_wfs_frame() rotates the launcher position and the beacon tilt
  in the same sense as the pupil rotation of the SH (EFInterpolator).
- LaserLaunchTelescope(absolute_coordinates=True) stores them in the telescope
  frame, and the SH gives its ConvolutionKernel the rotated ones.
"""
import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
from unittest import mock

from astropy.io import fits

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope, launcher_in_wfs_frame
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_noise_cov_elong import calc_noise_cov_elong
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.make_mask import make_mask
from specula.processing_objects.sh import SH
from test.specula_testlib import cpu_and_gpu


def _cov_full_grid(n, launcher, **kwargs):
    '''Inverse covariance (full matrix) of a full n x n grid, no mask'''
    args = dict(diameter_in_m=8., zenith_angle_in_deg=30., na_thickness_in_m=10e3,
                launcher_coord_in_m=launcher, sub_aps_index=np.arange(n * n),
                n_sub_aps=n, sub_aps_fov=5.0, sh_spot_fwhm=1.0, sigma_noise2=2.0,
                t_g_parameter=0.0, h_in_m=90e3)
    args.update(kwargs)
    return calc_noise_cov_elong(**args)


class TestSubApertureCentres(unittest.TestCase):

    def _precisions(self, n, launcher):
        cov = _cov_full_grid(n, launcher)
        d = np.diag(cov)
        # arrays indexed [row, col]
        return d[:n * n].reshape(n, n), d[n * n:].reshape(n, n)

    def test_mirror_symmetry_launcher_at_centre(self):
        for n in (4, 5):
            px, py = self._precisions(n, [0., 0., 0.])
            # x precision depends on the elongation along x: mirror in x
            np.testing.assert_allclose(px, px[:, ::-1], rtol=1e-9)
            # and the same for y
            np.testing.assert_allclose(py, py[::-1, :], rtol=1e-9)
            # Not trivially constant
            self.assertGreater(px.max() - px.min(), 1e-3 * px.max(), msg=f'n={n}')

    def test_odd_grid_middle_column_has_no_x_elongation(self):
        # Middle column/row: coordinate exactly 0, beta = 0, so
        # x diagonal = 1/sigma_noise2 * 1/(1+0) * (1 + beta2^2/s2) / (1 + beta2^2/s2)
        # = 1/sigma_noise2 exactly. Same for y in the middle row.
        n = 5
        px, py = self._precisions(n, [0., 0., 0.])
        np.testing.assert_allclose(px[:, n // 2], 1 / 2.0, rtol=1e-9)
        np.testing.assert_allclose(py[n // 2, :], 1 / 2.0, rtol=1e-9)
        # Off the middle column, the x precision is lower
        self.assertTrue(np.all(px[:, [0, 1, 3, 4]] < 1 / 2.0))

    def test_corner_convention_is_asymmetric(self):
        # Launcher shifted by +d/2 reproduces the old corner coordinates
        # (centre - d/2): the mirror symmetry must be lost.
        for n in (4, 5):
            half = 8. / n / 2
            px, py = self._precisions(n, [half, half, 0.])
            self.assertFalse(np.allclose(px, px[:, ::-1], rtol=1e-6), msg=f'n={n}')
            self.assertFalse(np.allclose(py, py[::-1, :], rtol=1e-6), msg=f'n={n}')


class TestLauncherInWfsFrame(unittest.TestCase):

    def test_quarter_turn(self):
        pos, tt = launcher_in_wfs_frame([1., 0., 7.], [1., 0.], 90.)
        np.testing.assert_allclose(pos, [0., 1., 7.], atol=1e-12)
        np.testing.assert_allclose(tt, [0., 1.], atol=1e-12)

    def test_quarter_turn_of_y_axis(self):
        pos, tt = launcher_in_wfs_frame([0., 1., 0.], [0., 1.], 90.)
        np.testing.assert_allclose(pos, [-1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(tt, [-1., 0.], atol=1e-12)

    def test_zero_is_identity(self):
        pos, tt = launcher_in_wfs_frame([3., -2., 5.], [0.3, 0.4], 0.)
        np.testing.assert_allclose(pos, [3., -2., 5.], atol=1e-15)
        np.testing.assert_allclose(tt, [0.3, 0.4], atol=1e-15)

    def test_inverse_rotation(self):
        pos0, tt0 = [3., -2., 5.], [0.3, 0.4]
        pos, tt = launcher_in_wfs_frame(pos0, tt0, 37.)
        pos, tt = launcher_in_wfs_frame(pos, tt, -37.)
        np.testing.assert_allclose(pos, pos0, atol=1e-12)
        np.testing.assert_allclose(tt, tt0, atol=1e-12)

    def test_norms_and_z_preserved(self):
        pos0, tt0 = [3., -2., 5.], [0.3, 0.4]
        pos, tt = launcher_in_wfs_frame(pos0, tt0, 123.4)
        self.assertAlmostEqual(np.hypot(pos[0], pos[1]), np.hypot(3., 2.), places=12)
        self.assertAlmostEqual(np.hypot(tt[0], tt[1]), 0.5, places=12)
        self.assertEqual(pos[2], 5.)

    def test_composition_adds_angles(self):
        pos, tt = launcher_in_wfs_frame([3., -2., 5.], [0.3, 0.4], 20.)
        pos, tt = launcher_in_wfs_frame(pos, tt, 25.)
        pos2, tt2 = launcher_in_wfs_frame([3., -2., 5.], [0.3, 0.4], 45.)
        np.testing.assert_allclose(pos, pos2, atol=1e-12)
        np.testing.assert_allclose(tt, tt2, atol=1e-12)


class TestRotationConventionMatchesPupilRotation(unittest.TestCase):

    @staticmethod
    def _feature_position(target_device_idx, xp, rot_deg, n=96, px=24., py=12.):
        '''
        Centroid [x (columns), y (rows)] relative to the array centre of a
        Gaussian blob at (px, py), after the EFInterpolator rotation.
        '''
        ef = ElectricField(n, n, 0.1, S0=1, target_device_idx=target_device_idx)
        c = (n - 1) / 2  # array centre, pixel-centre convention
        rows, cols = np.mgrid[0:n, 0:n]
        blob = np.exp(-((cols - c - px) ** 2 + (rows - c - py) ** 2) / (2 * 2.0 ** 2))
        ef.A[:] = xp.asarray(blob, dtype=ef.A.dtype)
        interp = EFInterpolator(ef, (n, n), rotAnglePhInDeg=rot_deg,
                                target_device_idx=target_device_idx)
        interp.interpolate()
        out = cpuArray(interp.interpolated_ef().A).astype(float)
        return (np.sum(out * (cols - c)) / np.sum(out),
                np.sum(out * (rows - c)) / np.sum(out))

    @cpu_and_gpu
    def test_launcher_rotates_with_pupil(self, target_device_idx, xp):
        # Feature at 1/4 of the pupil along x and 1/8 along y (pupil = 96 px)
        px, py = 24., 12.
        for rot in (30., -50.):
            x, y = self._feature_position(target_device_idx, xp, rot, px=px, py=py)
            expected = launcher_in_wfs_frame([px, py, 0.], [0., 0.], rot)[0][:2]
            self.assertAlmostEqual(x, expected[0], delta=0.3, msg=f'rot={rot}')
            self.assertAlmostEqual(y, expected[1], delta=0.3, msg=f'rot={rot}')

    @cpu_and_gpu
    def test_zero_rotation_keeps_position(self, target_device_idx, xp):
        # Needs force_extrapolation-free path: with rot=0 and same size no interpolation
        # is done, so use a tiny rotation to still go through Interp2D
        x, y = self._feature_position(target_device_idx, xp, 1e-3, px=24., py=12.)
        self.assertAlmostEqual(x, 24., delta=0.1)
        self.assertAlmostEqual(y, 12., delta=0.1)


class TestCovElongWfsRotation(unittest.TestCase):

    def test_matches_hand_rotated_launcher(self):
        n, rot = 4, 30.
        launcher = [3.0, -1.0, 0.]
        mask = make_mask(n, xp=np)
        idx = np.ravel_multi_index(np.where(mask), mask.shape)
        rotated, _ = launcher_in_wfs_frame(launcher, [0., 0.], rot)
        for only_diag in (False, True):
            ref = _cov_full_grid(n, rotated, sub_aps_index=idx, only_diag=only_diag)
            res = _cov_full_grid(n, launcher, sub_aps_index=idx, only_diag=only_diag,
                                 wfs_rotation_in_deg=rot)
            np.testing.assert_allclose(res, ref, rtol=1e-12)
            unrot = _cov_full_grid(n, launcher, sub_aps_index=idx, only_diag=only_diag)
            self.assertFalse(np.allclose(res, unrot, rtol=1e-3))

    def test_none_equals_no_rotation(self):
        launcher = [3.0, -1.0, 0.]
        np.testing.assert_array_equal(
            _cov_full_grid(4, launcher, wfs_rotation_in_deg=None),
            _cov_full_grid(4, launcher))
        # 0 deg is also the identity
        np.testing.assert_allclose(
            _cov_full_grid(4, launcher, wfs_rotation_in_deg=0.),
            _cov_full_grid(4, launcher), rtol=1e-12)

    def test_theta_is_rotated_with_lgs_map(self):
        # eta_is_not_one: the spots are computed by lgs_map_sh, with launcher and theta
        n, rot = 3, 40.
        launcher, theta = [2.0, 1.0, 0.], [0.5, -0.2]
        rot_launcher, rot_theta = launcher_in_wfs_frame(launcher, theta, rot)
        kw = dict(eta_is_not_one=True, only_diag=True)
        ref = _cov_full_grid(n, rot_launcher, theta=rot_theta, **kw)
        res = _cov_full_grid(n, launcher, theta=theta, wfs_rotation_in_deg=rot, **kw)
        np.testing.assert_allclose(res, ref, rtol=1e-10)
        # Rotating only the launcher (theta unrotated) gives a different result
        wrong = _cov_full_grid(n, rot_launcher, theta=theta, **kw)
        self.assertFalse(np.allclose(res, wrong, rtol=1e-6))


class TestLaserLaunchTelescopeAbsolute(unittest.TestCase):

    def setUp(self):
        self.simul_params = SimulParams(pixel_pupil=10, pixel_pitch=1.0, zenithAngleInDeg=30.0)

    def _llt(self, target_device_idx, absolute, tel_position=(2., 3., 4.)):
        return LaserLaunchTelescope(simul_params=self.simul_params, spot_size=1.0,
                                    tel_position=list(tel_position), beacon_tt=[5., 6.],
                                    absolute_coordinates=absolute,
                                    target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_default_is_not_absolute(self, target_device_idx, xp):
        llt = LaserLaunchTelescope(simul_params=self.simul_params, spot_size=1.0,
                                   tel_position=[1., 2., 3.], target_device_idx=target_device_idx)
        self.assertFalse(llt.absolute_coordinates)

    @cpu_and_gpu
    def test_wfs_frame_unchanged_if_not_absolute(self, target_device_idx, xp):
        llt = self._llt(target_device_idx, False)
        for rot in (0., 30., 90.):
            pos, tt = llt.wfs_frame(rot)
            self.assertEqual(pos, [2., 3., 4.])
            self.assertEqual(tt, [5., 6.])

    @cpu_and_gpu
    def test_wfs_frame_rotated_if_absolute(self, target_device_idx, xp):
        llt = self._llt(target_device_idx, True)
        pos, tt = llt.wfs_frame(90.)
        np.testing.assert_allclose(pos, [-3., 2., 4.], atol=1e-12)
        np.testing.assert_allclose(tt, [-6., 5.], atol=1e-12)
        ref = launcher_in_wfs_frame([2., 3., 4.], [5., 6.], 33.)
        res = llt.wfs_frame(33.)
        np.testing.assert_allclose(res[0], ref[0])
        np.testing.assert_allclose(res[1], ref[1])

    @cpu_and_gpu
    def test_wfs_frame_empty_position_unchanged(self, target_device_idx, xp):
        llt = self._llt(target_device_idx, True, tel_position=())
        pos, tt = llt.wfs_frame(45.)
        self.assertEqual(len(pos), 0)
        self.assertEqual(tt, [5., 6.])

    @cpu_and_gpu
    def test_header_roundtrip_keeps_flag(self, target_device_idx, xp):
        for absolute in (True, False):
            llt = self._llt(target_device_idx, absolute)
            hdr = llt.get_fits_header()
            self.assertEqual(hdr['ABSCOORD'], absolute)
            llt2 = LaserLaunchTelescope.from_header(hdr, target_device_idx=target_device_idx)
            self.assertEqual(llt2.absolute_coordinates, absolute)

    @cpu_and_gpu
    def test_file_roundtrip_keeps_flag(self, target_device_idx, xp):
        llt = self._llt(target_device_idx, True)
        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, 'llt.fits')
            llt.save(filename)
            self.assertTrue(LaserLaunchTelescope.restore(filename).absolute_coordinates)

    @cpu_and_gpu
    def test_old_header_without_flag_is_not_absolute(self, target_device_idx, xp):
        hdr = self._llt(target_device_idx, True).get_fits_header()
        del hdr['ABSCOORD']
        llt = LaserLaunchTelescope.from_header(hdr, target_device_idx=target_device_idx)
        self.assertFalse(llt.absolute_coordinates)


class TestSHLauncherFrame(unittest.TestCase):

    ROT = 30.

    def _sh(self, target_device_idx, absolute):
        llt = LaserLaunchTelescope(simul_params=SimulParams(zenithAngleInDeg=30.0),
                                   spot_size=1.0, tel_position=[5.0, 1.0, 0.0],
                                   beacon_tt=[0.5, -0.25], absolute_coordinates=absolute,
                                   target_device_idx=target_device_idx)
        ef = ElectricField(80, 80, 0.1, S0=100, target_device_idx=target_device_idx)
        sh = SH(wavelengthInNm=589, subap_wanted_fov=6, sensor_pxscale=0.5,
                subap_on_diameter=10, subap_npx=12, laser_launch_tel=llt,
                rotAnglePhInDeg=self.ROT, data_dir=self._data_dir,
                target_device_idx=target_device_idx)
        sh.inputs['in_ef'].set(ef)
        return sh

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._data_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _kernel_kwargs(self, sh):
        with mock.patch('specula.processing_objects.sh.ConvolutionKernel') as ck:
            sh.setup()
        ck.assert_called_once()
        return ck.call_args.kwargs

    @cpu_and_gpu
    def test_kernel_receives_rotated_launcher_if_absolute(self, target_device_idx, xp):
        kw = self._kernel_kwargs(self._sh(target_device_idx, True))
        pos, tt = launcher_in_wfs_frame([5., 1., 0.], [0.5, -0.25], self.ROT)
        np.testing.assert_allclose(kw['launcher_pos'], pos, atol=1e-12)
        np.testing.assert_allclose(kw['theta'], tt, atol=1e-12)
        self.assertFalse(np.allclose(kw['launcher_pos'], [5., 1., 0.]))

    @cpu_and_gpu
    def test_kernel_receives_unchanged_launcher_if_not_absolute(self, target_device_idx, xp):
        kw = self._kernel_kwargs(self._sh(target_device_idx, False))
        self.assertEqual(list(kw['launcher_pos']), [5., 1., 0.])
        self.assertEqual(list(kw['theta']), [0.5, -0.25])

    @cpu_and_gpu
    def test_rotation_change_not_allowed_if_absolute(self, target_device_idx, xp):
        sh = self._sh(target_device_idx, True)
        with mock.patch('specula.processing_objects.sh.ConvolutionKernel'):
            sh.setup()
        with self.assertRaises(NotImplementedError):
            sh.update_interpolator_parameters(rotAnglePhInDeg=self.ROT + 1)
        # Same rotation, or no rotation given: allowed
        sh.update_interpolator_parameters(rotAnglePhInDeg=self.ROT)
        sh.update_interpolator_parameters(xShiftPhInPixel=0.5)

    @cpu_and_gpu
    def test_rotation_change_allowed_if_not_absolute(self, target_device_idx, xp):
        sh = self._sh(target_device_idx, False)
        with mock.patch('specula.processing_objects.sh.ConvolutionKernel'):
            sh.setup()
        sh.update_interpolator_parameters(rotAnglePhInDeg=self.ROT + 1)


if __name__ == '__main__':
    unittest.main()
