import specula
specula.init(0)  # Default target device for initialization

import unittest
import pytest
import numpy as np
import matplotlib

# Use non-interactive backend to prevent windows from popping up during tests
matplotlib.use('Agg')

from specula.data_objects.ifunc import IFunc
from specula.lib.make_mask import make_mask
from specula.lib.plot_utils import display_ifunc_2d
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
