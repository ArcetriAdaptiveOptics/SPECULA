import specula
specula.init(0)  # Default target device for initialization

import unittest
import pytest
import numpy as np
import matplotlib
from matplotlib import colors as mcolors

# Use non-interactive backend to prevent windows from popping up during tests
matplotlib.use('Agg')

from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.lib.make_mask import make_mask
from specula.lib.plot_utils import display_ifunc_2d, display_mcao_geom
from test.specula_testlib import cpu_and_gpu


class TestPlotUtils(unittest.TestCase):
    """Test the display functions for IFunc objects."""

    def setUp(self):
        """Set up common test data (small dimensions to keep tests fast)"""
        self.dim = 32
        self.nmodes = 5
        self.mask = make_mask(self.dim)

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_display_ifunc_grid(self, target_device_idx, xp):
        """Test that the mosaic grid is generated with the correct dimensions."""
        ifunc = IFunc(type_str='zernike', mask=self.mask, nmodes=self.nmodes,
                      npixels=self.dim, target_device_idx=target_device_idx)

        n_raw_col = 3
        shape_out = display_ifunc_2d(ifunc, n_raw_col=n_raw_col, do_not_show_ticks=True,
                                     show_plot=False)

        # We expect a 2D array of size (n_raw_col * dim) x (n_raw_col * dim)
        expected_shape = (n_raw_col * self.dim, n_raw_col * self.dim)
        self.assertEqual(shape_out.shape, expected_shape)

        matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_display_ifunc_1d_vector(self, target_device_idx, xp):
        """Test the reconstruction of a single shape from a 1D vector of coefficients."""
        ifunc = IFunc(type_str='zernike', mask=self.mask, nmodes=self.nmodes,
                      npixels=self.dim, target_device_idx=target_device_idx)

        # 1D vector with 3 coefficients
        modal_vector = [1.0, -0.5, 0.2] 
        shape_out = display_ifunc_2d(ifunc, modal_vector=modal_vector,
                                     show_plot=False)

        # We expect a single frame with the same dimensions as the mask
        self.assertEqual(shape_out.shape, (self.dim, self.dim))

        # Verify that the output is not entirely composed of zeros
        self.assertTrue(np.any(shape_out != 0))

        matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_display_ifunc_2d_vector(self, target_device_idx, xp):
        """Test the reconstruction of multiple frames from a 2D matrix of coefficients."""
        ifunc = IFunc(type_str='zernike', mask=self.mask, nmodes=self.nmodes, 
                      npixels=self.dim, target_device_idx=target_device_idx)

        num_frames = 4
        num_modes_to_use = 3
        # 2D Matrix: (frames, modes)
        modal_vector = np.random.random((num_frames, num_modes_to_use))

        shape_out = display_ifunc_2d(ifunc, modal_vector=modal_vector,
                                     show_plot=False)

        # We expect a 3D cube: (X, Y, frames)
        self.assertEqual(shape_out.shape, (self.dim, self.dim, num_frames))

        matplotlib.pyplot.close('all')

    @pytest.mark.filterwarnings('ignore:.*FigureCanvasAgg is non-interactive.*:UserWarning')
    @pytest.mark.filterwarnings('ignore:.*Matplotlib is currently using agg*:UserWarning')
    @cpu_and_gpu
    def test_display_ifunc_grid_with_m2c_obj(self, target_device_idx, xp):
        """Test that a valid M2C object is accepted and applied."""
        ifunc = IFunc(type_str='zernike', mask=self.mask, nmodes=self.nmodes,
                      npixels=self.dim, target_device_idx=target_device_idx)

        # Identity M2C preserves the number of displayed modes while exercising the code path.
        m2c_matrix = xp.eye(self.nmodes)
        m2c_obj = M2C(m2c=m2c_matrix, target_device_idx=target_device_idx)

        n_raw_col = 2
        shape_out = display_ifunc_2d(ifunc, m2c_obj=m2c_obj, n_raw_col=n_raw_col,
                                     do_not_show_ticks=True, show_plot=False)

        expected_shape = (n_raw_col * self.dim, n_raw_col * self.dim)
        self.assertEqual(shape_out.shape, expected_shape)

        matplotlib.pyplot.close('all')

    def test_display_mcao_geom_conversion(self):
        """Test IDL-converted MCAO SA geometry plotting helper."""
        diam = 8.0
        no_subaps = 10
        no_lgs = 4
        lgs_height = 90000.0
        dm_height = 10000.0
        fov = 30.0
        shifts = np.zeros((2, no_lgs), dtype=float)
        rotations = np.linspace(0.0, 0.15, no_lgs)

        out = display_mcao_geom(
            diam=diam,
            no_subaps=no_subaps,
            no_gs=no_lgs,
            gs_height=lgs_height,
            dm_height=dm_height,
            gs_fov_diam_asec=fov,
            shifts=shifts,
            rotations=rotations,
            title='test',
            display_sa_lines=True,
            show_plot=False,
        )

        self.assertIn('centers', out)
        self.assertEqual(out['centers'].shape, (no_lgs, no_subaps, no_subaps, 2))
        self.assertTrue(np.all(np.isfinite(out['centers'])))
        self.assertGreater(out['meta_diam'], diam)
        self.assertGreater(out['subaps_size'], 0.0)

        matplotlib.pyplot.close('all')

    def test_display_mcao_geom_no_subaps(self):
        """Test MCAO geometry helper when no_subaps is None."""
        no_gs = 3
        out = display_mcao_geom(
            diam=8.0,
            no_gs=no_gs,
            gs_height=90000.0,
            dm_height=10000.0,
            gs_fov_diam_asec=30.0,
            shifts=np.zeros((2, no_gs), dtype=float),
            rotations=np.zeros(no_gs, dtype=float),
            no_subaps=None,
            show_plot=False,
        )

        self.assertIsNone(out['subaps_size'])
        self.assertIsNone(out['sa_shift'])
        self.assertIsNone(out['centers'])
        self.assertEqual(out['gs_centers'].shape, (no_gs, 2))
        self.assertTrue(np.all(np.isfinite(out['gs_centers'])))
        # One metapupil circle + one GS circle per guide star.
        self.assertEqual(len(out['axes'].patches), 1 + no_gs)
        # In this mode we draw circles, not scatter points.
        self.assertEqual(len(out['axes'].collections), 0)

        matplotlib.pyplot.close('all')

    def test_display_mcao_geom_no_subaps_filled_circles(self):
        """Test filled GS circles option when no_subaps is None."""
        no_gs = 2
        out = display_mcao_geom(
            diam=8.0,
            no_gs=no_gs,
            gs_height=90000.0,
            dm_height=10000.0,
            gs_fov_diam_asec=30.0,
            no_subaps=None,
            gs_circles_filled=True,
            show_plot=False,
        )

        # First patch is metapupil (not filled), subsequent ones are GS circles.
        self.assertEqual(len(out['axes'].patches), 1 + no_gs)
        for gs_patch in out['axes'].patches[1:]:
            self.assertTrue(gs_patch.get_fill())

        matplotlib.pyplot.close('all')

    def test_display_mcao_geom_pythonic_figsize_and_ax(self):
        """Test pythonic plotting parameters: figsize and external axes."""
        no_gs = 3
        fig, ax = matplotlib.pyplot.subplots(figsize=(6, 4))

        out = display_mcao_geom(
            diam=8.0,
            no_gs=no_gs,
            gs_height=90000.0,
            dm_height=10000.0,
            gs_fov_diam_asec=30.0,
            shifts=np.zeros((2, no_gs), dtype=float),
            rotations=np.zeros(no_gs, dtype=float),
            no_subaps=6,
            figsize=(5, 5),
            ax=ax,
            show_plot=False,
        )

        self.assertIs(out['figure'], fig)
        self.assertIs(out['axes'], ax)
        self.assertEqual(out['centers'].shape, (no_gs, 6, 6, 2))

        matplotlib.pyplot.close('all')

    def test_display_mcao_geom_extended_geometry_inputs(self):
        """Test newer geometry inputs aligned with compute_mcao_geom semantics."""
        out = display_mcao_geom(
            diam=39.0,
            no_gs=6,
            gs_height=np.inf,
            dm_height=15000.0,
            gs_fov_diam_asec=120.0,
            tech_fov_diam_asec=180.0,
            ngs_fov_diam_asec=30.0,
            sci_fov_diam_asec=40.0,
            sci_square=True,
            no_subaps=8,
            show_plot=False,
        )

        self.assertGreater(out['meta_diam'], 39.0)
        self.assertEqual(out['gs_patch_diam'], 39.0)  # gs_height = inf -> no cone shrink
        self.assertEqual(out['tech_fov_diam_asec'], 180.0)
        self.assertEqual(out['gs_angle_asec'], 60.0)
        self.assertIsNotNone(out['ngs_meta_diam'])
        self.assertIsNotNone(out['sci_meta_diam'])
        self.assertTrue(out['sci_square'])
        self.assertTrue(np.isfinite(out['gs_fov_dm']))

        matplotlib.pyplot.close('all')

    def test_display_mcao_geom_uniform_color_mode(self):
        """Test same-color mode for all GS with alpha for overlap visualization."""
        no_gs = 4
        out = display_mcao_geom(
            diam=8.0,
            no_gs=no_gs,
            gs_height=90000.0,
            dm_height=10000.0,
            gs_fov_diam_asec=30.0,
            no_subaps=None,
            gs_uniform_color=True,
            gs_color='tab:blue',
            gs_alpha=0.35,
            gs_circles_filled=True,
            show_plot=False,
        )

        gs_patches = out['axes'].patches[1:]
        self.assertEqual(len(gs_patches), no_gs)

        expected_edge = mcolors.to_rgba('tab:blue', alpha=0.35)
        for p in gs_patches:
            self.assertEqual(tuple(np.round(p.get_edgecolor(), 6)), tuple(np.round(expected_edge, 6)))

        matplotlib.pyplot.close('all')
